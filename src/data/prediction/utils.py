from typing import Any, Sequence, TypeGuard, cast
import logging
import random
from collections import namedtuple
from pydash import flatten
import numpy as np
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
import polars as pl

from core.ner.spacy import Spacy
from typings.core import Primitive
from utils.list import batch
from utils.tensor import batch_as_tensors, array_to_tensor

from .constants import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_STRING_LEN,
    DEFAULT_TEXT_FEATURES,
)


MAX_ITEMS_PER_CAT = 4

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


VectorizedFeatures = namedtuple(
    "VectorizedFeatures", ["encodings", "embeddings", "embedding_weights"]
)


def is_tensor_list(
    embeddings: list[torch.Tensor] | list[Primitive],
) -> TypeGuard[list[torch.Tensor]]:
    return isinstance(embeddings[0], torch.Tensor)


def batch_and_pad(
    tensors: list[torch.Tensor] | list[Primitive], batch_size: int
) -> torch.Tensor:
    """
    Batch a list of tensors or primitives

    Args:
        tensors (list[torch.Tensor] | list[Primitive]): List of tensors or primitives
        batch_size (int): Batch size
    """
    if not is_tensor_list(tensors):
        # then list of primitives
        batches = batch_as_tensors(cast(list[Primitive], tensors), batch_size)
        num_dims = 1
    else:
        num_dims = len(tensors[0].size()) + 1
        batches = batch(tensors, batch_size)
        batches = [torch.stack(b) for b in batches]

    def get_batch_pad(b: torch.Tensor):
        zeros = ([0, 0] * (num_dims - 1)) + [0]
        return (*zeros, batch_size - b.size(0))

    batches = [F.pad(b, get_batch_pad(b)) for b in batches]

    logging.info("Batches: %s (%s)", len(batches), [b.size() for b in batches])
    return torch.stack(batches)


def resize_and_batch(embeddings: list[torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    Size embeddings into batches

    Args:
        embeddings (list[torch.Tensor]): List of embeddings
        batch_size (int): Batch size
    """
    max_len = max(e.size(0) for e in embeddings)
    padded_emb = [F.pad(f, (0, max_len - f.size(0))) for f in embeddings]
    return batch_and_pad(padded_emb, batch_size)


def encode_category(field: str, df: pl.DataFrame) -> tuple[LabelEncoder, list[Any]]:
    """
    Encode a categorical field from a dataframe

    Args:
        field (str): Field to encode
        df (pl.DataFrame): Dataframe to encode from
    """
    encoder = LabelEncoder()
    new_list = df.select(pl.col(field)).to_series().to_list()
    logger.info("Encoding field %s (ex: %s)", field, new_list[0:5])
    encoder.fit(flatten(new_list))
    values = [
        encoder.transform([x] if isinstance(x, str) else x)
        for x in new_list
        if x is not None  # TODO: shouldn't be necessary; pre-clean
    ]
    logger.info("Finished encoding field %s", field)
    return (encoder, values)


def get_string_values(item: dict, field: str) -> list:
    val = item.get(field)
    if isinstance(val, list):
        return [v[0:DEFAULT_MAX_STRING_LEN] for v in val]
    if val is None:
        return [None]
    return [val[0:DEFAULT_MAX_STRING_LEN]]


def __get_text_embeddings(
    items: Sequence[dict],
    text_fields: list[str] = [],
) -> list[torch.Tensor]:
    """
    Get text embeddings given a list of dicts
    """
    nlp = Spacy.get_instance(disable=["ner"])
    pca = PCA(n_components=DEFAULT_TEXT_FEATURES)
    text_vectors = [
        np.concatenate(
            [
                tv
                for tv in [
                    np.array(
                        [
                            token.vector
                            for value in get_string_values(item, field)
                            for token in nlp(value)
                        ]
                    )
                    for field in text_fields
                ]
                if len(tv.shape) > 1
            ]
        )
        for item in items
    ]

    pca_model = pca.fit(np.concatenate(text_vectors, axis=0))
    text_feats = [
        torch.flatten(torch.tensor(pca_model.transform(text_vector)))
        for text_vector in text_vectors
    ]
    max_len = max(f.size(0) for f in flatten(text_feats))
    text_feats = [F.pad(f, (0, max_len - f.size(0))) for f in text_feats]
    return text_feats


def __vectorize_categories(
    items: Sequence[dict],
    categorical_fields: list[str],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> VectorizedFeatures:
    """
    Get category embeddings given a list of dicts
    """
    df = pl.from_dicts(items, infer_schema_length=1000)  # type: ignore
    # {'design': 5, 'masking': 5, 'randomization': 3, 'conditions': 1045, 'interventions': 1505, 'phase': 8}
    max_sixes = dict(
        [
            (field, df.select(pl.col(field).flatten()).n_unique())
            for field in categorical_fields
        ]
    )

    embedding_layers = [
        nn.Embedding(size, embedding_dim) for size in max_sixes.values()
    ]
    embedding_weights = [e.weight for e in embedding_layers]

    # one encoder per field
    label_encoders = [encode_category(field, df) for field in categorical_fields]
    FeatureTuple = namedtuple("FeatureTuple", categorical_fields)  # type: ignore

    # e.g. [{'conditions': array([413]), 'phase': array([2])}, {'conditions': array([436]), 'phase': array([2])}]
    encoded_dicts = [
        FeatureTuple(*fv)._asdict() for fv in zip(*[enc[1] for enc in label_encoders])
    ]

    # e.g. [[[413], [2]], [[436, 440], [2]]]
    encoded_records = [
        [dict[field][0:MAX_ITEMS_PER_CAT] for field in categorical_fields]
        for dict in encoded_dicts
    ]

    # torch.Size([2000, 6, 4])
    encodings = array_to_tensor(
        encoded_records, (*np.array(encoded_records).shape, MAX_ITEMS_PER_CAT)
    )

    embedded_records: list[list[torch.Tensor]] = [
        [
            torch.Tensor(embedding_layers[i](dict[i][0:MAX_ITEMS_PER_CAT]))
            for i in range(len(categorical_fields))
        ]
        for dict in encodings
    ]

    embeddings = array_to_tensor(
        embedded_records,
        (
            len(embedded_records),
            len(embedded_records[0]),
            *embedded_records[0][0].shape,
        ),
    )

    return VectorizedFeatures(
        encodings=encodings,
        embeddings=embeddings,
        embedding_weights=embedding_weights,
    )


def vectorize_features(
    records: Sequence[dict],
    categorical_fields: list[str],
    text_fields: list[str] = [],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> VectorizedFeatures:
    """
    Get category and text embeddings given a list of dicts

    Args:
        records (Sequence[dict]): List of dicts
        categorical_fields (list[str]): List of fields to embed as categorical variables
        text_fields (list[str]): List of fields to embed as text
        embedding_dim (int, optional): Embedding dimension. Defaults to EMBEDDING_DIM.
    """

    logger.info("Getting feature embeddings")
    # _records = [*records]
    # random.shuffle(_items)

    #  torch.Size([2000, 6, 3])
    vectorized_cats = __vectorize_categories(records, categorical_fields, embedding_dim)
    vectorized_text = (
        __get_text_embeddings(records, text_fields) if len(text_fields) > 1 else None
    )

    def get_all_features(i: int) -> torch.Tensor:
        if vectorized_text is not None:
            return torch.cat([vectorized_cats.embeddings[i], vectorized_text[i]], dim=0)
        else:
            return vectorized_cats.embeddings[i]

    embeddings = [get_all_features(i) for i, _ in enumerate(records)]

    logger.info("Finished with feature embeddings")
    return VectorizedFeatures(
        encodings=vectorized_cats.encodings,
        embedding_weights=vectorized_cats.embedding_weights,
        embeddings=embeddings,
    )
