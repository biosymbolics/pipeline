import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SOURCE_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations_source"
WORKING_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations"
AGGREGATED_ANNOTATIONS_TABLE = "aggregated_annotations"  # a mat view
REGULATORY_APPROVAL_TABLE = "regulatory_approvals"
PATENT_TO_REGULATORY_APPROVAL_TABLE = "patent_to_regulatory_approval"
PATENT_TO_TRIAL_TABLE = "patent_to_trial"
TRIALS_TABLE = "trials"
APPLICATIONS_TABLE = "applications"
PUBLICATION_NUMBER_MAP_TABLE = "publication_number_map"

ANNOTATIONS_TABLE = "annotations"
TERMS_TABLE = "terms"
TERM_IDS_TABLE = "term_ids"
SYNONYM_TABLE_NAME = "synonym_map"
COMPANIES_TABLE_NAME = "companies"


logger.info("Environment is: %s", os.environ.get("ENV", "local"))

IS_LOCAL = not os.environ.get("ENV") or os.environ.get("ENV") == "local"
IS_DEPLOYED = not IS_LOCAL
BASE_DATABASE_URL = "postgres://:@localhost:5432"
DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    if IS_DEPLOYED
    else "postgres://:@localhost:5432/patents"
)
DB_CLUSTER = os.environ.get("DB_CLUSTER")
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
