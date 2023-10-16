from datetime import date
from functools import reduce
from typing import Literal, NamedTuple, Sequence, TypeGuard, cast
import logging
from collections import namedtuple
from pydash import flatten
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer
from kneed import KneeLocator
import polars as pl

from core.ner.spacy import Spacy
from data.types import FieldLists, InputFieldLists
from typings.core import Primitive
from utils.list import batch
from utils.tensor import batch_as_tensors, array_to_tensor

from .constants import (
    DEFAULT_MAX_STRING_LEN,
    DEFAULT_TEXT_FEATURES,
)


DEFAULT_MAX_ITEMS_PER_CAT = 20

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CategorySizes(NamedTuple):
    multi_select: dict[str, int]
    single_select: dict[str, int]


# keep in sync with ModelInputAndOutput
class ModelInput(NamedTuple):
    multi_select: torch.Tensor
    quantitative: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor


# keep in sync with ModelInput
class ModelInputAndOutput(NamedTuple):
    multi_select: torch.Tensor
    quantitative: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor
    y1_categories: torch.Tensor  # used as y1_true (encodings)
    y1_true: torch.Tensor  # embedded y1_true
    y2_true: torch.Tensor
    y2_oh_true: torch.Tensor


class EncodedCategories(NamedTuple):
    category_size_map: dict[str, int]
    encodings: torch.Tensor


def is_tensor_list(
    embeddings: list[torch.Tensor] | list[Primitive],
) -> TypeGuard[list[torch.Tensor]]:
    return len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor)


def to_float(value: int | float | date) -> float:
    if isinstance(value, date):
        return float(value.year)
    return float(value)


def estimate_n_bins(
    data: list[str] | list[int] | list[float],
    kbins_strategy: Literal["uniform", "quantile", "kmeans"],
    bins_to_test=range(3, 20),
):
    """
    Estimate the optimal number of bins
    for use with bin_quantitative_values

    Calculates gini_impurity and uses elbow method to find optimal number of bins

    Args:
        data (list[str] | list[int] | list[float]): List of values
        bins_to_test (range): Range of bins to test
        kbins_strategy (str): Strategy to use for KBinsDiscretizer
    """

    def elbow(
        values: Sequence[float],
        bins: Sequence[int],
        strategy: Literal["first", "default"] = "first",
    ) -> int:
        # https://arvkevi-kneed.streamlit.app/
        # https://github.com/arvkevi/kneed
        kneedle = KneeLocator(
            bins, values, direction="decreasing", curve="concave", online=True
        )

        if kneedle.elbow is None:
            logger.warning("No elbow found for bins %s, using last index", bins)
            return len(bins) - 1

        if strategy == "first":
            return kneedle.all_elbows.pop()

        return int(kneedle.elbow)

    def gini_impurity(original, binned):
        hist, _ = np.histogram(original)
        gini_before = 1 - np.sum([p**2 for p in hist / len(original)])

        def bin_gini(bin):
            p = np.mean(binned == bin)
            return p * (1 - p)

        gini_after = reduce(lambda acc, bin: acc + bin_gini(bin), np.unique(binned))

        return gini_before - gini_after

    def score_bin(n_bins):
        est = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy=kbins_strategy, subsample=None
        )
        bin_data = est.fit_transform(np.array(data).reshape(-1, 1))
        return gini_impurity(data, bin_data)

    scores = [score_bin(n_bins) for n_bins in bins_to_test]
    winner = elbow(scores, bins_to_test)

    return winner


def bin_quantitative_values(
    values: Sequence[float | int] | pl.Series,
    field: str,
    n_bins: int | None = 5,
    kbins_strategy: Literal["uniform", "quantile", "kmeans"] = "kmeans",
) -> Sequence[list[int]]:
    """
    Bins quantiative values, turning them into categorical
    @see https://scikit-learn.org/stable/auto_examples/preprocessing/plot_discretization_strategies.html

    NOTE: specify n_bins when doing inference; i.e. ensure it matches with how the model was trained.

    Args:
        values (Sequence[float | int]): List of values
        field (str): Field name (used for logging)
        n_bins (int): Number of bins
        kbins_strategy (str): Strategy to use for KBinsDiscretizer

    Returns:
        Sequence[list[int]]: List of lists of binned values (e.g. [[0.0], [2.0], [5.0], [0.0], [0.0]])
            (a list of lists because that matches our other categorical vars)
    """
    if n_bins is None:
        n_bins = estimate_n_bins(list(values), kbins_strategy=kbins_strategy)
        logger.info(
            "Using estimated optimal n_bins value of %s for field %s", n_bins, field
        )

    binner = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy=kbins_strategy, subsample=None
    )
    X = np.array(values).reshape(-1, 1)
    Xt = binner.fit_transform(X)
    res = Xt.tolist()
    return res


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
        return torch.Tensor()  # return empty tensor

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
        batch_dim = (batch_size if batch_size > -1 else len(tensors)) - b.size(0)
        return (*zeros, batch_dim)

    batches = [F.pad(b, get_batch_pad(b)) for b in batches]

    logging.info("Batches: %s (%s ...)", len(batches), [b.size() for b in batches[0:5]])

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

    logger.info("Finished encoding field %s (e.g. %s)", field, encoded_values[0:5])
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
    device: str = "cpu",
) -> EncodedCategories:
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
        encoded_records,
        (len(records), len(categorical_fields), max_items_per_cat),
    ).to(device)

    return EncodedCategories(category_size_map=size_map, encodings=encodings)


def encode_single_select_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    device: str = "cpu",
) -> EncodedCategories:
    return encode_categories(
        records, categorical_fields, max_items_per_cat=1, device=device
    )


def encode_multi_select_categories(
    records: Sequence[dict],
    categorical_fields: list[str],
    max_items_per_cat: int,
    device: str = "cpu",
) -> EncodedCategories:
    return encode_categories(
        records,
        categorical_fields,
        max_items_per_cat=max_items_per_cat,
        device=device,
    )


def encode_quantitative_fields(
    records: Sequence[dict],
    fields: list[str],
) -> Sequence[dict]:
    """
    Encode quantitative fields into categorical
    (Intended for use as a preprocessing step, so that the
    newly categorical fields can be encoded as embeddings)

    Args:
        records (Sequence[dict]): List of dicts
        fields (list[str]): List of fields to encode

    Returns:
        Sequence[dict]: List of dicts with encoded fields
    """
    df = pl.from_dicts(records, infer_schema_length=1000)  # type: ignore
    for field in fields:
        df = df.with_columns(
            [
                pl.col(field).map_batches(
                    lambda v: pl.Series(bin_quantitative_values(v, field))
                )
            ]
        )
    return df.to_dicts()


def encode_features(
    records: Sequence[dict],
    field_lists: FieldLists | InputFieldLists,
    max_items_per_cat: int = DEFAULT_MAX_ITEMS_PER_CAT,
    device: str = "cpu",
) -> tuple[ModelInput, CategorySizes]:
    """
    Get category and text embeddings given a list of dicts

    Args:
        records (Sequence[dict]): List of dicts
        field_lists (FieldLists): Field lists
        max_items_per_cat (int): Max items per category
        device (str): Device to use
    """

    logger.info("Getting feature embeddings")

    single_select = encode_single_select_categories(
        records, field_lists.single_select, device=device
    )

    multi_select = encode_multi_select_categories(
        records,
        field_lists.multi_select,
        max_items_per_cat,
        device=device,
    )

    text = (
        __get_text_embeddings(records, field_lists.text, device)
        if len(field_lists.text) > 1
        else torch.Tensor()
    )

    quantitative = (
        F.normalize(
            torch.Tensor(
                [
                    [to_float(r[field]) for field in field_lists.quantitative]
                    for r in records
                ]
            ).to(device),
        )
        if len(field_lists.quantitative) > 0
        else torch.Tensor()
    )

    logger.info("Finished with feature encodings")

    return ModelInput(
        multi_select=multi_select.encodings,
        quantitative=quantitative,
        single_select=single_select.encodings,
        text=text,
    ), CategorySizes(
        multi_select=multi_select.category_size_map,
        single_select=single_select.category_size_map,
    )


def encode_and_batch_features(
    records: Sequence[dict],
    field_lists: FieldLists | InputFieldLists,
    batch_size: int,
    max_items_per_cat: int = DEFAULT_MAX_ITEMS_PER_CAT,
    device: str = "cpu",
) -> tuple[ModelInput, CategorySizes]:
    """
    Vectorizes and batches input features
    """
    feats, sizes = encode_features(
        records, field_lists, max_items_per_cat, device=device
    )

    feature_dict = dict(
        (f, resize_and_batch(t, batch_size)) for f, t in feats._asdict().items()
    )

    inputs = ModelInput(**feature_dict)

    logger.info(
        "Feature Sizes: %s",
        [(f, t.shape if t is not None else None) for f, t in inputs._asdict().items()],
    )

    return inputs, sizes
