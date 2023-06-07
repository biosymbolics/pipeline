from typing import Literal, Tuple
from llama_index.indices.base import BaseGPTIndex

LlmIndex = BaseGPTIndex  # TODO: abstract?
NamespaceKey = Tuple[str, ...]
LlmModel = Literal["ChatGPT", "VertexAI", "Anthropic"]
