import torch

from constants.patents import ATTRIBUTE_FIELD


LR = 1e-6  # learning rate
CHECKPOINT_PATH = "patent_model_checkpoints"
OPTIMIZER_CLASS = torch.optim.Adam
SAVE_FREQUENCY = 5
EMBEDDING_DIM = 16  # good? should be small stuff
MAX_STRING_LEN = 200
BATCH_SIZE = 256
TEXT_FEATURES = 20
MAX_CATS_PER_LIST = 20
TRUE_THRESHOLD = 0.5

CATEGORICAL_FIELDS = [
    ATTRIBUTE_FIELD,
    "compounds",
    "country",
    "diseases",
    "mechanisms",
    "ipc_codes",
]
TEXT_FIELDS = [
    "title",
    "abstract",
    # "claims",
    "assignees",
    "inventors",
]

GNN_CATEGORICAL_FIELDS = [
    "diseases",
    "compounds",
    "mechanisms",
]
