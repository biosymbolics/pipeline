from types import UnionType
from typing import NamedTuple, Sequence, Type, TypeGuard

from data.prediction.constants import (
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
MAX_ITEMS_PER_CAT = 10  # DEFAULT_MAX_ITEMS_PER_CAT
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    "enrollment",
    # "facilities", ??
    # "countries" ??
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",  # typically only the specific condition
    "mesh_conditions",  # normalized; includes ancestors
    "interventions",
]
TEXT_FIELDS: list[str] = []


QUANTITATIVE_FIELDS: list[str] = [
    "start_date",
]

# Side note: enrollment + duration have very low correlation, high covariance
# (low corr is perhaps why it doesn't offer much loss reduction)
QUANTITATIVE_TO_CATEGORY_FIELDS: list[str] = [
    "enrollment",
    "duration",
    # dropout_count
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
    "comparison_type",
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

output_field_lists = OutputFieldLists(
    y1_categorical=Y1_CATEGORICAL_FIELDS,
    y2=Y2_FIELD,
)


def _get_type(field_type: str) -> type | UnionType:
    if field_type == "single_select":
        return str | int | float
    if field_type == "multi_select":
        return list[str | int | float]
    if field_type == "text":
        return str
    if field_type == "quantitative":
        return int | float
    # TODO: hard-coded
    if field_type == "y1_categorical":
        return str
    if field_type == "y2":
        return int | float
    raise ValueError(f"Invalid field_type: {field_type}")


def get_fields_to_types(
    _field_lists: AnyFieldLists,
) -> tuple[tuple[str, type | UnionType], ...]:
    vals = _field_lists.__dict__.items()
    return tuple([(fv, _get_type(k)) for k, fvs in vals for fv in fvs])


# sigh https://github.com/python/mypy/issues/848
InputRecord = NamedTuple("InputRecord", get_fields_to_types(input_field_lists))  # type: ignore
InputAndOutputRecord = NamedTuple(  # type: ignore
    "InputAndOutputRecord", get_fields_to_types(field_lists)  # type: ignore
)
OutputRecord = NamedTuple("OutputRecord", get_fields_to_types(output_field_lists))  # type: ignore

AnyRecord = InputRecord | InputAndOutputRecord | OutputRecord


def is_output_record(record: AnyRecord) -> TypeGuard[OutputRecord]:
    return isinstance(record, OutputRecord)


def is_output_records(
    records: Sequence[AnyRecord],
) -> TypeGuard[Sequence[OutputRecord]]:
    return all(is_output_record(record) for record in records)
