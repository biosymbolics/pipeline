import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_NAME = "ChatGPT"

SOURCE_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations_source"
WORKING_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations"
AGGREGATED_ANNOTATIONS_TABLE = "aggregated_annotations"  # a mat view

logger.info("Environment is: %s", os.environ.get("ENV", "local"))

IS_LOCAL = os.environ.get("ENV") == "local"
IS_DEPLOYED = not IS_LOCAL
BASE_DATABASE_URL = "postgres://:@localhost:5432"
DATABASE_URL = (
    os.environ["DATABASE_URL"] if IS_DEPLOYED else "postgres://:@localhost:5432/patents"
)
DEFAULT_ENTITY_TYPES = frozenset(["compounds", "diseases", "mechanisms"])


DEFAULT_BASE_NLP_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEFAULT_NLP_DOC_STRIDE = 16
DEFAULT_NLP_MAX_LENGTH = 128  # same as with training
DEFAULT_TORCH_DEVICE: str = "mps"

DEFAULT_NLP_MODEL_ARGS = {
    "max_length": DEFAULT_NLP_MAX_LENGTH,
    "stride": DEFAULT_NLP_DOC_STRIDE,
    "return_tensors": "pt",
    "padding": "max_length",
    "truncation": True,
}
