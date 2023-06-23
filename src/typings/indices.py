from typing import Literal, NamedTuple
from llama_index import QuestionAnswerPrompt, RefinePrompt
from llama_index.indices.base import BaseIndex

LlmIndex = BaseIndex
NamespaceKey = NamedTuple
LlmModelType = Literal["ChatGPT", "VertexAI", "Anthropic"]


# adding these here in case we want to abstract 'em
Prompt = QuestionAnswerPrompt
RefinePrompt = RefinePrompt
