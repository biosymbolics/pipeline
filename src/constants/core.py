from clients.low_level.big_query import BQ_DATASET_ID
from typings.indices import LlmModelType


DEFAULT_MODEL_NAME: LlmModelType = "ChatGPT"

SOURCE_BIOSYM_ANNOTATIONS_TABLE = f"{BQ_DATASET_ID}.biosym_annotations"
WORKING_BIOSYM_ANNOTATIONS_TABLE = f"{BQ_DATASET_ID}.biosym_annotations_source"
