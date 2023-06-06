"""
Functions around llama index context
"""
from typing import Optional
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, VertexAI
import logging

from .types import LlmModel
from .utils import get_persist_dir


def get_storage_context(namespace: str) -> StorageContext:
    """
    Get storage context

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
    """
    directory = get_persist_dir(namespace)

    try:
        storage_context = StorageContext.from_defaults(persist_dir=directory)
    except Exception as ex:
        # assuming this means the directory does not exist
        # https://github.com/jerryjliu/llama_index/issues/3734
        logging.info(
            "Exception when loading storage context; assuming directory does not exist: %s",
            ex,
        )
        storage_context = StorageContext.from_defaults()
    return storage_context


def __get_llm(model_name: Optional[LlmModel]):
    """
    Get llm based on model_name

    Args:
        model_name (LlmModel): model to use for llm
    """
    common_args = {"temperature": 0.3}
    if model_name == "ChatGPT":
        return ChatOpenAI(
            **common_args, model="gpt-3.5-turbo", max_tokens=1900, client="chat"
        )
    if model_name == "VertexAI":
        return VertexAI(**common_args, model_name="text-bison")
    if model_name == "Anthropic":
        # untested, but used in https://colab.research.google.com/drive/1uuqvPI2_WNFMd7g-ahFoioSHV7ExB2GR?usp=sharing
        # benefit is massive input token limit
        # use with GPTListIndex?
        return Anthropic(**common_args, model="claude-v1.3-100k")

    raise Exception(f"Unknown model {model_name}")


def get_service_context(model_name: Optional[LlmModel] = "ChatGPT") -> ServiceContext:
    """
    Get default service context for llllamama index

    Args:
        model_name (Literal["ChatGPT", "VertexAI"]): model to use for llm
    """
    prompt_helper = PromptHelper(
        context_window=1024, num_output=256, chunk_overlap_ratio=0.1
    )

    llm = __get_llm(model_name)
    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return service_context
