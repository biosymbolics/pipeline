from types import UnionType
from typing import NamedTuple, Type

from data.prediction.constants import (
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)
from data.types import FieldLists, InputFieldLists

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


def get_input_type(field_type: str) -> Type | UnionType:
    if field_type == "single_select":
        return str | int | float
    if field_type == "multi_select":
        return list[str | int | float]
    if field_type == "text":
        return str
    if field_type == "quantitative":
        return int | float
    raise ValueError(f"Invalid field_type: {field_type}")


fields_to_types: list[tuple[str, Type | UnionType]] = [
    (fv, get_input_type(t))
    for t, fvs in input_field_lists._asdict().items()
    for fv in fvs
]
InputRecord = NamedTuple("InputRecord", fields_to_types)  # type: ignore

# with 5 buckets
# INFO:__main__:Training Stage1 design accuracy: 0.79
# INFO:__main__:Training Stage1 masking accuracy: 0.63
# INFO:__main__:Training Stage1 randomization accuracy: 0.81
# INFO:__main__:Training Stage1 comparison_type accuracy: 0.62
# INFO:__main__:Training Stage2 accuracy: 0.79
# INFO:__main__:Training Stage2 mae: 0.3
# INFO:__main__:Evaluation Stage1 design accuracy: 0.64
# INFO:__main__:Evaluation Stage1 masking accuracy: 0.5
# INFO:__main__:Evaluation Stage1 randomization accuracy: 0.69
# INFO:__main__:Evaluation Stage1 comparison_type accuracy: 0.43
# INFO:__main__:Evaluation Stage2 accuracy: 0.72
# INFO:__main__:Evaluation Stage2 mae: 0.37
