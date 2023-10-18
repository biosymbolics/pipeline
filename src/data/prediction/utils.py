from datetime import date
from typing import Any, NamedTuple, Sequence, TypeGuard, cast
import logging
from collections import namedtuple
import polars as pl
import torch
import torch.nn.functional as F
import polars as pl

from data.prediction.clindev.constants import InputRecord
from data.prediction.constants import DEFAULT_TEXT_FEATURES
from data.types import FieldLists, InputFieldLists
from typings.core import Primitive
from utils.encoding.quant_encoder import BinEncoder
from utils.encoding.saveable_encoder import LabelCategoryEncoder
from utils.encoding.text_encoder import get_text_embeddings
from utils.list import batch
from utils.tensor import batch_as_tensors, array_to_tensor


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

    logging.info(
        "Batches: %s (%s) (%s ...)",
        len(batches),
        batch_size,
        [b.size() for b in batches[0:5]],
    )

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


def is_list(obj: Any) -> bool:
    return hasattr(obj, "__len__") or isinstance(obj, list)


def encode_categories(
    records: Sequence[InputRecord],
    categorical_fields: list[str],
    max_items_per_cat: int,
    directory: str,
    device: str = "cpu",
) -> EncodedCategories:
    """
    Get category **encodings** given a list of dicts
    """
    dicts = [
        record._asdict() if not isinstance(record, dict) else record
        for record in records
    ]
    FeatureTuple = namedtuple("FeatureTuple", categorical_fields)  # type: ignore
    df = pl.from_dicts(dicts, infer_schema_length=1000)

    # total records by field
    size_map = dict(
        [
            (field, df.select(pl.col(field).flatten()).n_unique())
            for field in categorical_fields
        ]
    )

    # e.g. [{'conditions': array([413]), 'phase': array([2])}, {'conditions': array([436]), 'phase': array([2])}]
    encodings_list = [
        LabelCategoryEncoder(field, directory).fit_transform(df)
        for field in categorical_fields
    ]
    encoded_dicts = [FeatureTuple(*fv)._asdict() for fv in zip(*encodings_list)]

    # e.g. [[[413], [2]], [[436, 440], [2]]]
    encoded_records = [
        [
            dict[field][0:max_items_per_cat] if is_list(dict[field]) else [dict[field]]
            for field in categorical_fields
        ]
        for dict in encoded_dicts
    ]

    encodings = array_to_tensor(
        encoded_records,
        (len(records), len(categorical_fields), max_items_per_cat),
    ).to(device)

    return EncodedCategories(category_size_map=size_map, encodings=encodings)


def encode_single_select_categories(
    records: Sequence[InputRecord],
    categorical_fields: list[str],
    directory: str,
    device: str = "cpu",
) -> EncodedCategories:
    return encode_categories(
        records,
        categorical_fields,
        max_items_per_cat=1,
        directory=directory,
        device=device,
    )


def encode_multi_select_categories(
    records: Sequence[InputRecord],
    categorical_fields: list[str],
    max_items_per_cat: int,
    directory: str,
    device: str = "cpu",
) -> EncodedCategories:
    return encode_categories(
        records,
        categorical_fields,
        max_items_per_cat=max_items_per_cat,
        directory=directory,
        device=device,
    )


def encode_quantitative_fields(
    records: Sequence[dict],
    fields: list[str],
    directory: str,
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
                    lambda v: pl.Series(BinEncoder(field, directory).bin(v))
                )
            ]
        )
    return df.to_dicts()


def encode_features(
    records: Sequence[InputRecord],
    field_lists: FieldLists | InputFieldLists,
    directory: str,
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
        records, field_lists.single_select, directory=directory, device=device
    )

    multi_select = encode_multi_select_categories(
        records,
        field_lists.multi_select,
        max_items_per_cat,
        directory=directory,
        device=device,
    )

    text = (
        get_text_embeddings(
            records,
            field_lists.text,
            n_text_features=DEFAULT_TEXT_FEATURES,
            device=device,
        )
        if len(field_lists.text) > 1
        else torch.Tensor()
    )

    quantitative = (
        F.normalize(
            torch.Tensor(
                [
                    [to_float(r[field]) for field in field_lists.quantitative]  # type: ignore
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
    records: Sequence[InputRecord],
    field_lists: FieldLists | InputFieldLists,
    batch_size: int,
    directory: str,
    max_items_per_cat: int = DEFAULT_MAX_ITEMS_PER_CAT,
    device: str = "cpu",
) -> tuple[ModelInput, CategorySizes]:
    """
    Vectorizes and batches input features
    """
    feats, sizes = encode_features(
        records, field_lists, directory, max_items_per_cat, device=device
    )

    batched = dict(
        (f, resize_and_batch(t, batch_size)) for f, t in feats._asdict().items()
    )

    logger.info(
        "Feature Sizes: %s",
        [(f, t.shape if t is not None else None) for f, t in batched.items()],
    )

    return ModelInput(**batched), sizes
