from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from langchain import OpenAI

from .utils import get_persist_dir

def get_storage_context(namespace: str) -> StorageContext:
    """
    Get storage context

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
    """
    directory = get_persist_dir(namespace)
    storage_context = StorageContext.from_defaults(persist_dir=directory)
    return storage_context

def get_service_context():
    """
    Get default service context for llllamama index
    """
    max_input_size = 4096
    num_output = 2048
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    llm_predictor = LLMPredictor(
        llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1900)
    )

    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return service_context
