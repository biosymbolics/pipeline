"""
Functions around llama index context
"""
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from langchain.chat_models import ChatOpenAI
import logging

# from langchain import OpenAI

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


def get_service_context():
    """
    Get default service context for llllamama index
    """
    max_input_size = 4096
    num_output = 3072  # 2048
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0, model="gpt-3.5-turbo", max_tokens=1900, client="chat"
        )
        # llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1000)
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return service_context
