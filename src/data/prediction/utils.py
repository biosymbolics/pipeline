from typing import Any, Iterable, NamedTuple, Sequence, TypeGuard, cast
import logging
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


MAX_ITEMS_PER_CAT = 20

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VectorizedCategories(NamedTuple):
    category_size_map: dict[str, int]
    encodings: torch.Tensor
    embeddings: torch.Tensor
    weights: list[torch.Tensor]


class VectorizedFeatures(NamedTuple):
    multi_select: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor
    quantitative: torch.Tensor


class EmbeddedCategories(NamedTuple):
    embeddings: torch.Tensor
    weights: list[torch.Tensor]


def is_tensor_list(
    embeddings: list[torch.Tensor] | list[Primitive],
) -> TypeGuard[list[torch.Tensor]]:
    return len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor)


def batch_and_pad(
    tensors: list[torch.Tensor] | list[Primitive], batch_size: int
) -> torch.Tensor:
    """
    Batch a list of tensors or primitives

    Args:
        tensors (list[torch.Tensor] | list[Primitive]): List of tensors or primitives
        batch_size (int): Batch size
    """
    if len(tensors) == 0:
        return torch.empty(batch_size, 0)  # return batch-sized but empty tensor

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


def resize_and_batch(
    tensors: list[torch.Tensor] | torch.Tensor, batch_size: int
) -> torch.Tensor:
    """
    Size embeddings into batches

    Args:
        tensors (list[torch.Tensor]): List of tensors
        batch_size (int): Batch size
    """
    sizes = [e.size(0) for e in tensors if len(e.size()) > 1]

    if len(sizes) > 0:
        max_len = max(sizes)
        padded_emb = [F.pad(f, (0, max_len - f.size(0))) for f in tensors]
    else:
        padded_emb = [f for f in tensors]
    return batch_and_pad(padded_emb, batch_size)


def encode_category(field: str, df: pl.DataFrame) -> list[list[int]]:
    """
    Encode a categorical field from a dataframe

    Args:
        field (str): Field to encode
        df (pl.DataFrame): Dataframe to encode from
    """
    values = df.select(pl.col(field)).to_series().to_list()
    logger.info(
        "Encoding field %s (e.g.: %s) length: %s", field, values[0:5], len(values)
    )
    encoder = LabelEncoder()
    # flatten list values; only apply here otherwise value error
    encoder.fit(flatten(values))
    encoded_values = [
        encoder.transform([v] if isinstance(v, str) else v) for v in values
    ]

    if df.shape[0] != len(encoded_values):
        raise ValueError(
            "Encoded values length does not match dataframe length: %s != %s",
            len(encoded_values),
            df.shape[0],
        )

    logger.info("Finished encoding field %s (%s)", field, encoded_values[0])
    return cast(list[list[int]], encoded_values)


def get_string_values(item: dict, field: str) -> list:
    val = item.get(field)
    if isinstance(val, list):
        return [v[0:DEFAULT_MAX_STRING_LEN] for v in val]
    if val is None:
        return [None]
    return [val[0:DEFAULT_MAX_STRING_LEN]]


def __get_text_embeddings(
    records: Sequence[dict],
    text_fields: list[str] = [],
    device: str = "cpu",
) -> torch.Tensor:
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
                            for value in get_string_values(record, field)
                            for token in nlp(value)
                        ]
                    )
                    for field in text_fields
                ]
                if len(tv.shape) > 1
            ]
        )
        for record in records
    ]

    pca_model = pca.fit(np.concatenate(text_vectors, axis=0))
    text_feats = [
        torch.flatten(torch.tensor(pca_model.transform(text_vector)))
        for text_vector in text_vectors
    ]
    max_len = max(f.size(0) for f in flatten(text_feats))
    text_feats = [F.pad(f, (0, max_len - f.size(0))) for f in text_feats]

    tensor = array_to_tensor(
        text_feats,
        (
            len(text_feats),
            len(text_feats[0]),
            *text_feats[0][0].shape,
        ),
    )

    return tensor.to(device)


def encode_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    max_items_per_cat: int,
):
    """
    Get category **encodings** given a list of dicts
    """
    FeatureTuple = namedtuple("FeatureTuple", categorical_fields)  # type: ignore
    df = pl.from_dicts(records, infer_schema_length=1000)  # type: ignore

    size_map = dict(
        [
            (field, df.select(pl.col(field).flatten()).n_unique())
            for field in categorical_fields
        ]
    )

    # e.g. [{'conditions': array([413]), 'phase': array([2])}, {'conditions': array([436]), 'phase': array([2])}]
    encodings_list = [encode_category(field, df) for field in categorical_fields]
    encoded_dicts = [FeatureTuple(*fv)._asdict() for fv in zip(*encodings_list)]

    # e.g. [[[413], [2]], [[436, 440], [2]]]
    encoded_records = [
        [dict[field][0:max_items_per_cat] for field in categorical_fields]
        for dict in encoded_dicts
    ]

    encodings = array_to_tensor(
        encoded_records, (len(records), len(categorical_fields), max_items_per_cat)
    )

    return encodings, size_map


def __embed_categories(
    encodings: torch.Tensor,
    categorical_fields: list[str],
    max_sizes: dict[str, int],
    max_items_per_cat: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> EmbeddedCategories:
    """
    Get category embeddings given a list of dicts

    Usage:
    ```
    encodings, max_sizes = encode_categories(items, categorical_fields)
    __embed_categories(encodings, categorical_fields, max_sizes)
    ```
    """

    embedding_layers = [nn.Embedding(s, embedding_dim) for s in max_sizes.values()]
    weights = [e.weight for e in embedding_layers]

    embedded_records: list[list[torch.Tensor]] = [
        [
            torch.Tensor(embedding_layers[i](enc[i][0:max_items_per_cat]))
            for i in range(len(categorical_fields))
        ]
        for enc in encodings
    ]

    embeddings = array_to_tensor(
        embedded_records,
        (
            len(embedded_records),
            len(embedded_records[0]),
            *embedded_records[0][0].shape,
        ),
    ).to(device)

    return EmbeddedCategories(embeddings=embeddings, weights=weights)


def __vectorize_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    max_items_per_cat: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> VectorizedCategories:
    encodings, max_sizes = encode_categories(
        records, categorical_fields, max_items_per_cat
    )

    embeddings, weights = __embed_categories(
        encodings,
        categorical_fields,
        max_sizes,
        max_items_per_cat=max_items_per_cat,
        embedding_dim=embedding_dim,
        device=device,
    )

    return VectorizedCategories(
        category_size_map=max_sizes,
        encodings=encodings,
        embeddings=embeddings,
        weights=weights,
    )


def vectorize_single_select_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> VectorizedCategories:
    return __vectorize_categories(
        records,
        categorical_fields,
        max_items_per_cat=1,
        embedding_dim=embedding_dim,
        device=device,
    )


def vectorize_multi_select_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    max_items_per_cat: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> VectorizedCategories:
    return __vectorize_categories(
        records,
        categorical_fields,
        max_items_per_cat=max_items_per_cat,
        embedding_dim=embedding_dim,
        device=device,
    )


def vectorize_features(
    records: Sequence[dict],
    single_select_categorical_fields: list[str],
    multi_select_categorical_fields: list[str],
    text_fields: list[str] = [],
    quantitative_fields: list[str] = [],
    max_items_per_cat: int = MAX_ITEMS_PER_CAT,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    device: str = "cpu",
) -> VectorizedFeatures:
    """
    Get category and text embeddings given a list of dicts

    Args:
        records (Sequence[dict]): List of dicts
        single_select_categorical_fields (list[str]): List of fields to embed as single select categorical variables
        multi_select_categorical_fields (list[str]): List of fields to embed as multi select categorical variables
        text_fields (list[str]): List of fields to embed as text
        quantitative_fields (list[str]): List of fields to embed as quantitative variables
        embedding_dim (int, optional): Embedding dimension. Defaults to EMBEDDING_DIM.
    """

    logger.info("Getting feature embeddings")
    # _records = [*records]
    # random.shuffle(_items)

    vectorized_single_select = vectorize_single_select_categories(
        records, single_select_categorical_fields, embedding_dim, device=device
    )

    vectorized_multi_select = vectorize_multi_select_categories(
        records,
        multi_select_categorical_fields,
        max_items_per_cat,
        embedding_dim,
        device=device,
    )

    vectorized_text = (
        __get_text_embeddings(records, text_fields, device)
        if len(text_fields) > 1
        else torch.Tensor()
    )

    vectorized_quantitative = (
        F.normalize(
            torch.Tensor([[r[field] for field in quantitative_fields] for r in records])
        )
        if len(quantitative_fields) > 0
        else torch.Tensor()
    )

    logger.info("Finished with feature embeddings")
    return VectorizedFeatures(
        multi_select=vectorized_multi_select.embeddings,
        single_select=vectorized_single_select.embeddings,
        text=vectorized_text,
        quantitative=vectorized_quantitative,
    )


def vectorize_and_batch_features(
    batch_size: int,
    records: Sequence[dict],
    single_select_categorical_fields: list[str],
    multi_select_categorical_fields: list[str],
    text_fields: list[str] = [],
    quantitative_fields: list[str] = [],
    max_items_per_cat: int = MAX_ITEMS_PER_CAT,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    flatten_batch: bool = False,
    device: str = "cpu",
) -> VectorizedFeatures:
    """
    Vectorizes and batches input features
    """
    feats = vectorize_features(
        records,
        single_select_categorical_fields,
        multi_select_categorical_fields,
        text_fields,
        quantitative_fields,
        max_items_per_cat,
        embedding_dim,
        device=device,
    )

    feature_dict = dict(
        (f, resize_and_batch(t, batch_size)) for f, t in feats._asdict().items()
    )

    for f, t in feature_dict.items():
        logger.info("Feature %s: %s", f, t.size())

    # optionally turn into (seq length) x (everything else)
    if flatten_batch:
        feature_dict = dict(
            (f, t.view(*t.shape[0:2], -1) if len(t.shape) > 2 else t)
            for f, t in feature_dict.items()
        )

    batched = VectorizedFeatures(**feature_dict)

    logger.info(
        "Feature Sizes: %s",
        [(f, t.shape if t is not None else None) for f, t in batched._asdict().items()],
    )

    return batched
