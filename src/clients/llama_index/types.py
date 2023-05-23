from typing import TypeVar
from llama_index.indices.base import BaseGPTIndex as LlmIndex

LI = TypeVar("LI", bound=LlmIndex)
