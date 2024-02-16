import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPLICATIONS_TABLE = "applications"
GPR_ANNOTATIONS_TABLE = "gpr_annotations"
SOURCE_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations_source"
WORKING_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations"
PATENT_VECTOR_TABLE = "patent_vectors"
REGULATORY_APPROVAL_VECTOR_TABLE = "trial_vectors"
TRIAL_VECTOR_TABLE = "trial_vectors"
REGULATORY_APPROVAL_TABLE = "regulatory_approval"
SEARCH_TABLE = "doc_entity_search"
TRIALS_TABLE = "trial"


logger.info("Environment is: %s", os.environ.get("ENV", "local"))

IS_LOCAL = not os.environ.get("ENV") or os.environ.get("ENV") == "local"
IS_DEPLOYED = not IS_LOCAL
ETL_BASE_DATABASE_URL = "postgres://localhost:5432"
BASE_DATABASE_URL = "postgres://biosym:ok@localhost:5432"
DATABASE_URL = (
    os.environ.get("DATABASE_URL") if IS_DEPLOYED else f"{BASE_DATABASE_URL}/biosym"
)
DB_CLUSTER = os.environ.get("DB_CLUSTER")


DEFAULT_BASE_TRANSFORMER_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEFAULT_NLP_DOC_STRIDE = 16
DEFAULT_NLP_MAX_LENGTH = 512  # same as with training
DEFAULT_TORCH_DEVICE: str = os.environ.get("DEFAULT_TORCH_DEVICE") or "mps"

DEFAULT_NLP_MODEL_ARGS = {
    "max_length": DEFAULT_NLP_MAX_LENGTH,
    "stride": DEFAULT_NLP_DOC_STRIDE,
    "return_tensors": "pt",
    "padding": "max_length",
    "truncation": True,
}
