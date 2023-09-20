from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = DEFAULT_BATCH_SIZE
LR = DEFAULT_LR
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = 1  # DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
    "phase",
    # "comparator",
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = SINGLE_SELECT_CATEGORICAL_FIELDS
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",
    "interventions",
    # "termination_reason",
]
TEXT_FIELDS: list[str] = []
Y2_FIELD = "duration"
