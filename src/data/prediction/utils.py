from datetime import date
from functools import partial
from typing import Any, NamedTuple, Sequence, TypeGuard, TypeVar, cast
import logging
from collections import namedtuple
import polars as pl
from pydash import uniq
import torch
import torch.nn.functional as F
import polars as pl

from data.prediction.clindev.constants import (
    AnyRecord,
    InputAndOutputRecord,
    InputRecord,
    OutputRecord,
    is_output_records,
)
from data.prediction.constants import DEFAULT_TEXT_FEATURES
from data.prediction.types import (
    AllCategorySizes,
    CategorySizes,
    InputCategorySizes,
    ModelInput,
    ModelInputAndOutput,
    ModelOutput,
    OutputCategorySizes,
)
from typings.core import Primitive
from utils.encoding.quant_encoder import BinEncoder
from utils.encoding.saveable_encoder import LabelCategoryEncoder
from utils.encoding.text_encoder import TextEncoder
from utils.list import batch
from utils.tensor import batch_as_tensors, array_to_tensor

from .types import FieldLists, InputFieldLists, OutputFieldLists, is_all_fields_list


DEFAULT_MAX_ITEMS_PER_CAT = 20

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T", bound=AnyRecord)


class EncodedCategories(NamedTuple):
    category_size_map: dict[str, int]
    encodings: torch.Tensor


def is_tensor_list(
    embeddings: Sequence[torch.Tensor] | Sequence[Primitive],
) -> TypeGuard[Sequence[torch.Tensor]]:
    return len(embeddings) > 0 and isinstance(embeddings[0], torch.Tensor)


def to_float(value: int | float | date) -> float:
    if isinstance(value, date):
        return float(value.year)
    return float(value)


def batch_and_pad(
    tensors: Sequence[torch.Tensor] | Sequence[Primitive], batch_size: int
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
        batches = [torch.stack(list(b)) for b in batches]

    def get_batch_pad(b: torch.Tensor):
        zeros = ([0, 0] * (num_dims - 1)) + [0]
        batch_dim = (batch_size if batch_size > -1 else len(tensors)) - b.size(0)
        return (*zeros, batch_dim)

    batches = [F.pad(b, get_batch_pad(b)) for b in batches]

    logging.debug(
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
    records: Sequence[T],
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


def encode_single_select(
    records: Sequence[T],
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


def encode_multi_select(
    records: Sequence[T],
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
                    lambda v: pl.Series(BinEncoder(field, directory).fit_transform(v))
                )
            ]
        )
    return df.to_dicts()


def encode_outputs(
    records: Sequence[OutputRecord],
    field_lists: OutputFieldLists,
    directory: str,
    device: str = "cpu",
):
    """
    Encode outputs (for use in loss calculation)
    """
    encode_single = partial(encode_single_select, directory=directory, device=device)
    y1 = encode_single(records, field_lists.y1_categorical)
    y2 = encode_single(records, [field_lists.y2])
    y2_encoding = y2.encodings.squeeze(1)
    y2_size = y2.category_size_map[field_lists.y2]
    y2_oh = (
        torch.zeros(y2_encoding.size(0), y2_size).to(device).scatter_(1, y2_encoding, 1)
    )

    sizes = OutputCategorySizes(
        y1=y1.category_size_map,
        y2=y2_size,
    )
    encodings = ModelOutput(
        y1_true=y1.encodings,
        y2_true=y2_encoding,
        y2_oh_true=y2_oh,
    )
    return encodings, sizes


def encode_quantitative(
    records: Sequence[dict],
    fields: list[str],
    device: str = "cpu",
) -> torch.Tensor:
    return F.normalize(
        torch.Tensor([[to_float(r[field]) for field in fields] for r in records]),
    ).to(device)


def encode_features(
    records: Sequence[T],
    field_lists: FieldLists | InputFieldLists,
    directory: str,
    max_items_per_cat: int = DEFAULT_MAX_ITEMS_PER_CAT,
    device: str = "cpu",
) -> tuple[ModelInput, CategorySizes | AllCategorySizes]:
    """
    Get category and text embeddings given a list of dicts

    Args:
        records (Sequence[dict]): List of dicts
        field_lists (FieldLists): Field lists
        directory (str): Directory to save encoders
        max_items_per_cat (int): Max items per category
        device (str): Device to use
    """

    args = {"directory": directory, "device": device}

    field_types = {
        "multi_select": partial(
            encode_multi_select, max_items_per_cat=max_items_per_cat, **args
        ),
        "quantitative": partial(encode_quantitative, device=device),
        "single_select": partial(encode_single_select, **args),
        "text": partial(
            lambda records, f: TextEncoder(
                f, n_features=DEFAULT_TEXT_FEATURES, device=device
            ).fit_transform(pl.DataFrame(records, infer_schema_length=1000))
        ),
    }

    encoded = {
        t: enc_fun(records, getattr(field_lists, t))
        if len(getattr(field_lists, t)) > 0
        else torch.Tensor()
        for t, enc_fun in field_types.items()
    }

    sizes = InputCategorySizes(
        **{
            t: v.category_size_map
            for t, v in encoded.items()
            if isinstance(v, EncodedCategories)
        }
    )

    input_enc = ModelInput.get_instance(
        **{
            t: (v.encodings if isinstance(v, EncodedCategories) else v)
            for t, v in encoded.items()
        }
    )

    if is_all_fields_list(field_lists) and is_output_records(records):
        o_encodings, o_sizes = encode_outputs(
            records,
            cast(OutputFieldLists, field_lists),
            directory=directory,
            device=device,
        )
        # TODO: Ugly
        return ModelInputAndOutput.get_instance(
            **input_enc.__dict__, **o_encodings.__dict__
        ), AllCategorySizes(**sizes.__dict__, **o_sizes.__dict__)

    return input_enc, sizes


def decode_output(
    y1_probs_list: list[torch.Tensor],
    y2_preds: torch.Tensor,
    field_lists: OutputFieldLists,
    directory: str,
) -> dict[str, Any]:
    """
    Decode outputs

    TODO: generalize or move
    """
    y1_values = [torch.argmax(y1).item() for y1 in y1_probs_list]
    y2_pred = torch.argmax(torch.softmax(y2_preds, dim=1)).item()

    y1_decoded = {
        f: LabelCategoryEncoder(f, directory).inverse_transform([v])
        for f, v in zip(field_lists.y1_categorical, y1_values)
    }
    y2_decoded = LabelCategoryEncoder(field_lists.y2, directory).inverse_transform(
        [y2_pred]
    )
    y2_quant_decoded = BinEncoder(field_lists.y2, directory).inverse_transform(
        [y2_decoded]
    )
    return {**y1_decoded, field_lists.y2: round(y2_quant_decoded[0][0])}


def _encode_and_batch_features(
    records: Sequence[AnyRecord],
    field_lists: FieldLists | InputFieldLists,
    batch_size: int,
    directory: str,
    max_items_per_cat: int = DEFAULT_MAX_ITEMS_PER_CAT,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Vectorizes and batches input features
    """
    feats, sizes = encode_features(
        records, field_lists, directory, max_items_per_cat, device=device
    )

    batched = dict(
        (f, resize_and_batch(t, batch_size)) for f, t in feats.__dict__.items()
    )

    logger.debug(
        "Feature Sizes: %s",
        [(f, t.shape if t is not None else None) for f, t in batched.items()],
    )

    return batched, sizes.__dict__


def encode_and_batch_input(
    records: Sequence[InputRecord], field_lists: InputFieldLists, **kwargs
) -> tuple[ModelInput, InputCategorySizes]:
    batched, sizes = _encode_and_batch_features(records, field_lists, **kwargs)
    return ModelInput(**batched), InputCategorySizes(**sizes)


def encode_and_batch_all(
    records: Sequence[InputAndOutputRecord], field_lists: FieldLists, **kwargs
) -> tuple[ModelInputAndOutput, AllCategorySizes]:
    batched, sizes = _encode_and_batch_features(records, field_lists, **kwargs)
    return ModelInputAndOutput(**batched), AllCategorySizes(**sizes)


def split_train_and_test(
    input_dict: ModelInputAndOutput,
    training_proportion: float,
) -> tuple[ModelInputAndOutput, ModelInputAndOutput]:
    """
    Split out training and test data

    Args:
        input_dict (ModelInputAndOutput): Input data
        training_proportion (float): Proportion of data to use for training
    """
    record_cnt = input_dict.y1_true.size(0)
    split_idx = round(record_cnt * training_proportion)

    # len(v) == 0 if empty input
    split_input = {
        k: torch.split(v, split_idx) if len(v) > 0 else (torch.Tensor(), torch.Tensor())
        for k, v in input_dict.__dict__.items()
    }

    for i in range(2):
        sizes = uniq([len(v[i]) for v in split_input.values() if len(v[i]) > 0])
        if len(sizes) > 1:
            raise ValueError(
                f"Split discrepancy: {[(k, len(v[i])) for k, v in split_input.items()]}"
            )

    train_input_dict = ModelInputAndOutput(**{k: v[0] for k, v in split_input.items()})
    test_input = ModelInputAndOutput(**{k: v[1] for k, v in split_input.items()})

    return train_input_dict, test_input
