"""
Llama Index Types
"""
from typing import Literal, TypeVar
from llama_index.indices.base import BaseGPTIndex as LlmIndex

LI = TypeVar("LI", bound=LlmIndex)

LlmModel = Literal["ChatGPT", "VertexAI", "Anthropic"]
