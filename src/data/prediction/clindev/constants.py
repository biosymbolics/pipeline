from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


BATCH_SIZE = DEFAULT_BATCH_SIZE
EMBEDDING_DIM = 16
LR = 1e-3  # no learning at 1e-7, 1e-6
OPTIMIZER_CLASS = DEFAULT_OPTIMIZER_CLASS
SAVE_FREQUENCY = 1  # DEFAULT_SAVE_FREQUENCY
TRUE_THRESHOLD = DEFAULT_TRUE_THRESHOLD

SINGLE_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "phase",
    "sponsor_type",
    # "comparator",
    # "facilities", ??
    # "countries" ??
]
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",
    "interventions",
    # "termination_reason",
]
TEXT_FIELDS: list[str] = []
QUANTITATIVE_FIELDS: list[str] = [
    "enrollment",
]
Y2_FIELD = "duration"


# INFO:root:Stage 1 MSE: 77.90966796875 80
# INFO:root:Stage 2 MAE: 114.396240234375 115
# INFO:root:Stage 1 MSE: 102.35175323486328
# INFO:root:Stage 2 MAE: 115.066650390625
# INFO:root:Stage 1 MSE: 96.32074737548828
# INFO:root:Stage 2 MAE: 121.72994995117188
# INFO:root:Stage 1 MSE: 63.16200256347656
# INFO:root:Stage 2 MAE: 120.18370056152344
