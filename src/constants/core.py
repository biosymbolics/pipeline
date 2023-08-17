import logging
import os
from typings.indices import LlmModelType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_NAME: LlmModelType = "ChatGPT"

SOURCE_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations_source"
WORKING_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations"

logger.info("Envinroment is: %s", os.environ.get("ENV", "unknown"))

IS_LOCAL = os.environ.get("ENV") == "local"
IS_DEPLOYED = not IS_LOCAL
DATABASE_URL = (
    os.environ["DATABASE_URL"]
    if not IS_LOCAL
    else "postgres://:@localhost:5432/patents"
)
