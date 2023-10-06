from data.prediction.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OPTIMIZER_CLASS,
    DEFAULT_SAVE_FREQUENCY,
    DEFAULT_TRUE_THRESHOLD,
)

CHECKPOINT_PATH = "clindev_model_checkpoints"


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
Y1_CATEGORICAL_FIELDS: list[str] = [
    "design",
    "masking",
    "randomization",
    "comparison_type",
    # "hypothesis_type"
    # "termination_reason",
    # dropout_count
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
Y2_FIELD = "duration"

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
