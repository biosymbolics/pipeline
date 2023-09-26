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
    "comparator_type",
]
MULTI_SELECT_CATEGORICAL_FIELDS: list[str] = [
    "conditions",  # typically only the specific condition
    "mesh_conditions",  # normalized; includes ancestors
    "interventions",
    # "termination_reason",
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
]
Y2_FIELD = "duration"  # TODO: include distance measure


# lower LR (1e-3), no dropout
# INFO:root:Stage1 design Metrics: {'precision': 0.6007328782992817, 'recall': 0.5247468413229283, 'f1-score': 0.5450438363078526}
# INFO:root:Stage1 design Accuracy: 0.6814516129032258
# INFO:root:Stage1 masking Metrics: {'precision': 0.3934667220999648, 'recall': 0.3331720780161459, 'f1-score': 0.32092576577565735}
# INFO:root:Stage1 masking Accuracy: 0.5430443548387097
# INFO:root:Stage1 randomization Metrics: {'precision': 0.5752425923905847, 'recall': 0.43675346910274143, 'f1-score': 0.45396077515158284}
# INFO:root:Stage1 randomization Accuracy: 0.777016129032258
# INFO:root:Stage2 MAE: 155.0238029233871
# INFO:root:Saved checkpoint checkpoint_495.pt

# LR = 1e-4 no dropout early stopping
# INFO:root:Stage1 design Metrics: {'precision': 0.9534018783108008, 'recall': 0.9447741797276228, 'f1-score': 0.9490162443608342}
# INFO:root:Stage1 design Accuracy: 0.9602822580645162
# INFO:root:Stage1 masking Metrics: {'precision': 0.656033849922445, 'recall': 0.6207700903378501, 'f1-score': 0.6239451844747066}
# INFO:root:Stage1 masking Accuracy: 0.8418346774193548
# INFO:root:Stage1 randomization Metrics: {'precision': 0.8552756044490288, 'recall': 0.7361480323623094, 'f1-score': 0.7572132673289986}
# INFO:root:Stage1 randomization Accuracy: 0.9251008064516129
# INFO:root:Stage2 MAE: 62.788142641129035
