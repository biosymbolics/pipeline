from types import UnionType
from typing import NamedTuple, Sequence, TypeGuard
from pydash import flatten

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
MAX_ITEMS_PER_CAT = 8  # DEFAULT_MAX_ITEMS_PER_CAT
MAX_TOKENS_PER_ITEM = 8
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

TRAINING_PROPORTION = 0.8

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    "enrollment",
    # "facilities", ??
    # "countries" ??
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    # "conditions",  # typically only the specific condition
    # "mesh_conditions",  # normalized; includes ancestors
    # "interventions",
]
TEXT_FIELDS: list[str] = [
    # "conditions",  # typically only the specific condition
    "mesh_conditions",  # normalized; includes ancestors
    "interventions",
]


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
        return list[str] | list[int] | list[float]
    if field_type == "text":
        return str | list[str]
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
    record_keys = (record.__dict__ if not isinstance(record, dict) else record).keys()
    output_fields = set(flatten(output_field_lists.__dict__.values()))
    return output_fields.issubset(record_keys)


def is_output_records(
    records: Sequence[AnyRecord],
) -> TypeGuard[Sequence[OutputRecord]]:
    return len(records) > 0 and all(
        is_output_record(record) for record in records[0:10]
    )


# Epoch 50
# INFO:__main__:Training Stage1 design accuracy: 0.74
# INFO:__main__:Training Stage1 masking accuracy: 0.62
# INFO:__main__:Training Stage1 randomization accuracy: 0.83
# INFO:__main__:Training Stage1 comparison_type accuracy: 0.67
# INFO:__main__:Training Stage2 accuracy: 0.83
# INFO:__main__:Training Stage2 precision: 0.69
# INFO:__main__:Training Stage2 recall: 0.31
# INFO:__main__:Training Stage2 mae: 0.69
# INFO:__main__:Evaluation Stage1 design accuracy: 0.64
# INFO:__main__:Evaluation Stage1 masking accuracy: 0.53
# INFO:__main__:Evaluation Stage1 randomization accuracy: 0.76
# INFO:__main__:Evaluation Stage1 comparison_type accuracy: 0.55
# INFO:__main__:Evaluation Stage2 accuracy: 0.81
# INFO:__main__:Evaluation Stage2 precision: 0.55
# INFO:__main__:Evaluation Stage2 recall: 0.23
# INFO:__main__:Evaluation Stage2 mae: 0.86


# INFO:__main__:Starting epoch 20
# INFO:__main__:Training Stage1 design accuracy: 0.99
# INFO:__main__:Training Stage1 masking accuracy: 0.98
# INFO:__main__:Training Stage1 randomization accuracy: 0.99
# INFO:__main__:Training Stage1 comparison_type accuracy: 0.99
# INFO:__main__:Training Stage2 accuracy: 1.0
# INFO:__main__:Training Stage2 precision: 0.99
# INFO:__main__:Training Stage2 recall: 0.99
# INFO:__main__:Training Stage2 mae: 0.01
# INFO:__main__:Evaluation Stage1 design accuracy: 0.69
# INFO:__main__:Evaluation Stage1 masking accuracy: 0.48
# INFO:__main__:Evaluation Stage1 randomization accuracy: 0.76
# INFO:__main__:Evaluation Stage1 comparison_type accuracy: 0.56
# INFO:__main__:Evaluation Stage2 accuracy: 0.77
# INFO:__main__:Evaluation Stage2 precision: 0.41
# INFO:__main__:Evaluation Stage2 recall: 0.36
# INFO:__main__:Evaluation Stage2 mae: 0.94
