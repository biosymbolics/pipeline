from enum import Enum
from types import UnionType
from typing import NamedTuple, Sequence, TypeGuard
from pydash import flatten

from ..constants import (
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)
from ..types import AnyFieldLists, FieldLists, InputFieldLists, OutputFieldLists

CHECKPOINT_PATH = "clindev_model_checkpoints"
BASE_ENCODER_DIRECTORY = "clindev_model_checkpoints/encoders"
BATCH_SIZE = 32  # DEFAULT_BATCH_SIZE
DEVICE = "mps"
EMBEDDING_DIM = 16
LR = 1e-4
MAX_ITEMS_PER_CAT = 5  # DEFAULT_MAX_ITEMS_PER_CAT
MAX_TOKENS_PER_ITEM = 8
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

TRAINING_PROPORTION = 0.8

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    # "sponsor_type",
    "max_timeframe",
    # "dropout_count",
    # "facilities", ??
    # "countries" ??
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = []
TEXT_FIELDS: list[str] = ["indications", "interventions", "sponsor", "title"]


QUANTITATIVE_FIELDS: list[str] = [
    "start_date",
]

# Side note: enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_TO_CATEGORY_FIELDS: list[str] = [
    "enrollment",
    "duration",
    "max_timeframe",
    # "dropout_count",
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "blinding",
    "randomization",
    "comparison_type",
    "enrollment",
    # "max_timeframe",
    # "hypothesis_type"
    # "termination_reason",
    # dropout_count
]
Y2_FIELD = "duration"

field_lists = FieldLists(
    single_select=SINGLE_SELECT_CATEGORICAL_FIELDS,
    multi_select=MULTI_SELECT_CATEGORICAL_FIELDS,
    text=TEXT_FIELDS,
    quantitative=QUANTITATIVE_FIELDS,
    y1_categorical=Y1_CATEGORICAL_FIELDS,
    y2=Y2_FIELD,
)

input_field_lists = InputFieldLists(
    single_select=SINGLE_SELECT_CATEGORICAL_FIELDS,
    multi_select=MULTI_SELECT_CATEGORICAL_FIELDS,
    text=TEXT_FIELDS,
    quantitative=QUANTITATIVE_FIELDS,
)

ALL_INPUT_FIELD_LISTS: list[str] = flatten(input_field_lists.__dict__.values())

ALL_FIELD_LISTS: list[str] = flatten(field_lists.__dict__.values())

output_field_lists = OutputFieldLists(
    y1_categorical=Y1_CATEGORICAL_FIELDS,
    y2=Y2_FIELD,
)


def _get_type(field_type: str) -> UnionType:
    if field_type == "single_select":
        return str | int | float | Enum
    if field_type == "multi_select":
        return list[str] | list[int] | list[float] | list[Enum]
    if field_type == "text":
        return str | list[str]
    if field_type == "quantitative":
        return int | float
    # TODO: hard-coded
    if field_type == "y1_categorical":
        return str | str
    if field_type == "y2":
        return int | float
    raise ValueError(f"Invalid field_type: {field_type}")


def get_fields_to_types(
    _field_lists: AnyFieldLists,
) -> tuple[tuple[str, type | UnionType], ...]:
    vals = _field_lists.__dict__.items()
    vals = flatten(  # type: ignore
        [
            (
                [(v, _get_type(k))]
                if isinstance(v, str)
                else list(zip(v, [_get_type(k)] * len(v)))
            )
            for k, v in vals
        ]
    )
    return tuple(vals)


# sigh https://github.com/python/mypy/issues/848
# https://github.com/python/mypy/issues/6063
InputRecord = NamedTuple("InputRecord", get_fields_to_types(input_field_lists))  # type: ignore
InputAndOutputRecord = NamedTuple(  # type: ignore
    "InputAndOutputRecord", get_fields_to_types(field_lists)  # type: ignore
)
OutputRecord = NamedTuple("OutputRecord", get_fields_to_types(output_field_lists))  # type: ignore
AnyRecord = InputRecord | InputAndOutputRecord | OutputRecord


def is_output_record(record: AnyRecord) -> TypeGuard[OutputRecord]:
    record_keys = (record._asdict() if not isinstance(record, dict) else record).keys()
    output_fields = set(flatten(output_field_lists.__dict__.values()))
    return output_fields.issubset(record_keys)


def is_output_records(
    records: Sequence[AnyRecord],
) -> TypeGuard[Sequence[OutputRecord]]:
    return len(records) > 0 and all(
        is_output_record(record) for record in records[0:10]
    )
