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

from .constants import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_STRING_LEN,
    DEFAULT_TEXT_FEATURES,
)


from typings.core import Primitive
from utils.list import batch
from utils.tensor import batch_as_tensors

MAX_ITEMS_PER_CAT = 20

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        if num_dims == 1:
            return (0, batch_size - b.size(0))

        if num_dims == 2:
            return (0, 0, 0, batch_size - b.size(0))

        raise ValueError("Unsupported number of dimensions: %s" % num_dims)

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


def __get_cat_embeddings(
    items: Sequence[dict],
    categorical_fields: list[str],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> list[torch.Tensor]:
    """
    Get category embeddings given a list of dicts
    """
    df = pl.from_dicts(items, infer_schema_length=1000)  # type: ignore
    count_map = dict(
        [
            (field, df.select(pl.col(field).flatten()).n_unique())
            for field in categorical_fields
        ]
    )

    embedding_layers = dict(
        [
            (field, nn.Embedding(count_map[field], embedding_dim))
            for field in categorical_fields
        ]
    )

    label_encoders = [encode_category(field, df) for field in categorical_fields]
    CatTuple = namedtuple("CatTuple", categorical_fields)  # type: ignore
    cat_feats = [
        CatTuple(*fv)._asdict() for fv in zip(*[enc[1] for enc in label_encoders])
    ]

    cat_tensors: list[list[torch.Tensor]] = [
        [
            embedding_layers[field](
                torch.tensor(patent[field][0:MAX_ITEMS_PER_CAT], dtype=torch.int)
            )
            for field in categorical_fields
        ]
        for patent in cat_feats
    ]
    max_len_0 = max(f.size(0) for f in flatten(cat_tensors))
    max_len_1 = max(f.size(1) for f in flatten(cat_tensors))
    padded_cat_tensors = [
        [
            F.pad(f, (0, max_len_1 - f.size(1), 0, max_len_0 - f.size(0)))
            if f.size(0) > 0
            else torch.empty([max_len_0, max_len_1])
            for f in cat_tensor
        ]
        for cat_tensor in cat_tensors
    ]
    cat_feats = [torch.cat(tensor_set) for tensor_set in padded_cat_tensors]
    return cat_feats


def get_feature_embeddings(
    items: Sequence[dict],
    categorical_fields: list[str],
    text_fields: list[str] = [],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
):
    """
    Get category and text embeddings given a list of dicts

    Args:
        items (Sequence[dict]): List of dicts
        categorical_fields (list[str]): List of fields to embed as categorical variables
        text_fields (list[str]): List of fields to embed as text
        embedding_dim (int, optional): Embedding dimension. Defaults to EMBEDDING_DIM.
    """

    logger.info("Getting feature embeddings")
    _items = [*items]
    random.shuffle(_items)

    cat_feats = __get_cat_embeddings(_items, categorical_fields, embedding_dim)
    text_feats = (
        __get_text_embeddings(items, text_fields) if len(text_fields) > 1 else []
    )

    def get_all_features(i: int) -> torch.Tensor:
        combo_text_feats = (
            torch.flatten(text_feats[i]) if len(text_fields) > 0 else None
        )
        combo_cat_feats = torch.flatten(cat_feats[i])
        if combo_text_feats is not None:
            return torch.cat([combo_cat_feats, combo_text_feats], dim=0)
        else:
            return combo_cat_feats

    embeddings = [get_all_features(i) for i, _ in enumerate(items)]

    logger.info("Finished with feature embeddings")
    return embeddings
