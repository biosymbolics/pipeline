from datetime import date
from functools import partial
from typing import (
    Any,
    Callable,
    Mapping,
    NamedTuple,
    Sequence,
    TypeGuard,
    TypeVar,
    cast,
)
import logging
from collections import namedtuple
import polars as pl
import numpy.typing as npt
from pydash import is_list
import torch
import torch.nn.functional as F


from typings.core import Dataclass, Primitive
from typings.documents.trials import ScoredTrial
from utils.encoding.quant_encoder import BinEncoder
from utils.encoding.saveable_encoder import LabelCategoryEncoder
from utils.encoding.text_encoder import WORD_VECTOR_LENGTH, TextEncoder
from utils.list import batch
from utils.tensor import batch_as_tensors, array_to_tensor

from .clindev.constants import (
    AnyRecord,
    InputAndOutputRecord,
    InputRecord,
    OutputRecord,
    is_output_records,
)
from .types import (
    AllCategorySizes,
    CategorySizes,
    FieldLists,
    InputFieldLists,
    OutputFieldLists,
    is_all_fields_list,
    InputCategorySizes,
    ModelInput,
    ModelInputAndOutput,
    ModelOutput,
    OutputCategorySizes,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T", bound=AnyRecord)

GetEncoder = Callable[[str, str, int], Any]


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
        [b.size() for b in batches[0:6]],
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


def encode_categories(
    records: Sequence[T],
    fields: Sequence[str],
    max_items_per_cat: int,
    directory: str,
    get_encoder: GetEncoder = lambda field, directory, max_items_per_cat: LabelCategoryEncoder(
        field, directory
    ),
    device: str = "cpu",
) -> EncodedCategories:
    """
    Get category **encodings** given a list of dicts
    """

    df = pl.from_records(records)

    # total records by field
    exploded_df = df.with_columns(pl.col(pl.List).explode())
    size_map: dict[str, int] = dict(
        [
            (
                field,
                exploded_df.select(pl.col(field)).n_unique(),
            )
            for field in fields
        ]
    )

    # e.g. [{'conditions': array([413]), 'phase': array([2])}, {'conditions': array([436]), 'phase': array([2])}]
    encodings_list = [
        get_encoder(field, directory, max_items_per_cat).fit_transform(df)
        for field in fields
    ]
    encoded_dicts = [
        namedtuple("FeatureTuple", fields)(*fv)._asdict() for fv in zip(*encodings_list)  # type: ignore
    ]

    # e.g. [[[413], [2]], [[436, 440], [2]]]
    encoded_records = [
        [
            # TODO: not ideal to force list of scalar value
            dict[field][0:max_items_per_cat] if is_list(dict[field]) else [dict[field]]
            for field in fields
        ]
        for dict in encoded_dicts
    ]

    encodings = array_to_tensor(
        encoded_records,
        (len(records), len(fields), max_items_per_cat),
    ).to(device)

    return EncodedCategories(category_size_map=size_map, encodings=encodings)


def encode_categories_as_text(
    records: Sequence[T],
    fields: list[str],
    max_items_per_cat: int,
    max_tokens_per_item: int,
    directory: str,
    get_encoder: GetEncoder = (lambda f, dir, max_in_cat: TextEncoder(f, max_in_cat)),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Get vectorized text
    """

    dicts = [
        record._asdict() if not isinstance(record, dict) else record
        for record in records
    ]
    df = pl.from_dicts(dicts, infer_schema_length=1000)

    encodings_list = [
        get_encoder(field, directory, max_items_per_cat).fit_transform(df)
        for field in fields
    ]
    encodings = array_to_tensor(
        encodings_list,
        (
            len(records),
            len(fields),
            max_items_per_cat,
            max_tokens_per_item,
            WORD_VECTOR_LENGTH,
        ),
    ).to(device)

    return encodings.view(*encodings.shape[0:1], -1)


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
    records: Sequence[T],
    fields: Sequence[str],
    directory: str,
) -> list[T]:
    """
    Encode quantitative fields into categorical
    (Intended for use as a preprocessing step, so that the
    newly categorical fields can be encoded as embeddings)

    Args:
        records (Sequence[dict]): List of dicts
        fields (list[str]): List of fields to encode
        directory (str): Directory to save encoder

    Returns:
        Sequence[dict]: List of dicts with encoded fields
    """
    _records = [*records]

    def encode(field) -> list[T]:
        col_vals = [r.__dict__[field] for r in records]
        encoded = BinEncoder(field, directory).fit_transform(col_vals)
        return [r.replace(**{field: e}) for r, e in zip(_records, encoded)]

    for field in fields:
        _records = encode(field)

    return _records


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
    records: Sequence[T],
    fields: list[str],
    device: str = "cpu",
) -> torch.Tensor:
    return F.normalize(
        torch.Tensor(
            [[to_float(r._asdict()[field]) for field in fields] for r in records]
        ),
    ).to(device)


def encode_features(
    records: Sequence[T],
    field_lists: FieldLists | InputFieldLists,
    directory: str,
    max_items_per_cat: int,
    max_tokens_per_item: int,
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

    field_types: dict[str, Callable] = {
        "multi_select": partial(
            encode_multi_select, max_items_per_cat=max_items_per_cat, **args
        ),
        "quantitative": partial(encode_quantitative, device=device),
        "single_select": partial(encode_single_select, **args),
        "text": partial(
            encode_categories_as_text,
            **args,
            max_items_per_cat=max_items_per_cat,
            max_tokens_per_item=max_tokens_per_item,
        ),
    }

    encoded = {
        t: (
            Encode(records, getattr(field_lists, t))
            if len(getattr(field_lists, t)) > 0
            else torch.Tensor()
        )
        for t, Encode in field_types.items()
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

    # return encoded outputs too, if appropriate
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
    # temporary - truncate outputs to actual length, instead of returning values generated by padding
    actual_length: int | None = None,
) -> dict[str, npt.NDArray]:
    """
    Decode outputs

    TODO: generalize or move
    """
    length = actual_length or len(y1_probs_list)
    y1_values = [torch.argmax(y1, dim=1)[0:length] for y1 in y1_probs_list]
    y2_pred = torch.argmax(torch.softmax(y2_preds, dim=1), dim=1)[0:length]

    y1_decoded = {
        f: LabelCategoryEncoder(f, directory).inverse_transform(
            v.detach().cpu().numpy()
        )
        for f, v in zip(field_lists.y1_categorical, y1_values)
    }
    y2_decoded = LabelCategoryEncoder(field_lists.y2, directory).inverse_transform(
        y2_pred.detach().cpu().numpy()
    )
    y2_quant_decoded = (
        BinEncoder(field_lists.y2, directory)
        .inverse_transform(y2_decoded.reshape(-1, 1))
        .reshape(-1)
    )
    return {
        **y1_decoded,
        field_lists.y2: y2_decoded,
        f"{field_lists.y2}_exact": y2_quant_decoded,
    }


def _encode_and_batch_features(
    records: Sequence[AnyRecord],
    field_lists: FieldLists | InputFieldLists,
    batch_size: int,
    directory: str,
    max_items_per_cat: int,
    max_tokens_per_item: int,
    device: str = "cpu",
) -> tuple[dict, dict]:
    """
    Vectorizes and batches input features
    """
    feats, sizes = encode_features(
        records,
        field_lists,
        directory,
        max_items_per_cat,
        max_tokens_per_item,
        device=device,
    )

    batched = dict((f, resize_and_batch(t, batch_size)) for f, t in feats.items())

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


R = TypeVar("R", bound=Mapping | Dataclass | ScoredTrial)


def split_train_and_test(
    batched_inputs: ModelInputAndOutput,
    batched_records: Sequence[Sequence[R]],
    training_proportion: float,
) -> tuple[
    ModelInputAndOutput,
    ModelInputAndOutput,
    Sequence[Sequence[R]],
    Sequence[Sequence[R]],
]:
    """
    Split out training and test data

    Args:
        batched_inputs (ModelInputAndOutput): Batched input data
        training_proportion (float): Proportion of data to use for training
    """
    record_cnt = batched_inputs.y1_true.size(0)
    split_idx = round(record_cnt * training_proportion)

    # len(v) == 0 if empty input
    split_input = {
        k: torch.split(v, split_idx) if len(v) > 0 else (torch.Tensor(), torch.Tensor())
        for k, v in batched_inputs.items()
    }

    train_input_dict = ModelInputAndOutput(**{k: v[0] for k, v in split_input.items()})
    test_input = ModelInputAndOutput(**{k: v[1] for k, v in split_input.items()})

    return (
        train_input_dict,
        test_input,
        batched_records[0:split_idx],
        batched_records[split_idx:],
    )
