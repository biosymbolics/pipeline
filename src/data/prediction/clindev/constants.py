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
SAVE_FREQUENCY = DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "intervention_model",
    "masking",
    "randomization",
    # "comparator",
    # "facilities", ??
    # "countries" ??
]
CATEGORICAL_FIELDS: list[str] = [
    *Y1_CATEGORICAL_FIELDS,  # stage 1 predictions assist in stage 2 duration prediction
    "conditions",
    "interventions",
    "phase",
    "termination_reason",
]
TEXT_FIELDS: list[str] = []
Y2_FIELD = "duration"
