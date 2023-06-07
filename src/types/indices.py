from typing import Literal, Tuple
from llama_index import QuestionAnswerPrompt, RefinePrompt
from llama_index.indices.base import BaseGPTIndex

LlmIndex = BaseGPTIndex  # TODO: abstract?
NamespaceKey = Tuple[str, ...]
LlmModelType = Literal["ChatGPT", "VertexAI", "Anthropic"]


# adding these here in case we want to abstract 'em
Prompt = QuestionAnswerPrompt
RefinePrompt = RefinePrompt
