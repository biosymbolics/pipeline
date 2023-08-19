import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MODEL_NAME = "ChatGPT"

SOURCE_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations_source"
WORKING_BIOSYM_ANNOTATIONS_TABLE = "biosym_annotations"

logger.info("Environment is: %s", os.environ.get("ENV", "local"))

IS_LOCAL = os.environ.get("ENV") == "local"
IS_DEPLOYED = not IS_LOCAL
DATABASE_URL = (
    os.environ["DATABASE_URL"] if IS_DEPLOYED else "postgres://:@localhost:5432/patents"
)
