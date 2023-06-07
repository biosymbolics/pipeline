"""
Functions around llama index context
"""
from typing import Any, Literal, Mapping, NamedTuple, Optional
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, VertexAI
import logging

from constants.core import DEFAULT_MODEL_NAME
from clients.vector_dbs.pinecone import get_vector_db
from types.indices import LlmModelType, NamespaceKey

from .utils import get_persist_dir

DEFAULT_PINECONE_INDEX = "biosymbolics"


def __load_storage_context(**kwargs) -> StorageContext:
    """
    Load storage context
    """
    storage_context = None
    try:
        storage_context = StorageContext.from_defaults(**kwargs)
    except Exception as ex:
        # assuming this means the directory does not exist
        # https://github.com/jerryjliu/llama_index/issues/3734
        logging.info(
            "Exception when loading storage context; assuming directory does not exist: %s",
            ex,
        )
        storage_context = StorageContext.from_defaults()
    return storage_context


def get_storage_context(
    namespace_key: NamespaceKey,
    store_type: Optional[Literal["directory", "pinecone"]] = "directory",
    vector_store_kwargs: Mapping[str, Any] = {},
) -> StorageContext:
    """
    Get storage context

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        store_type (Literal["directory", "pinecone"]): type of storage to use
        vector_store_kwargs (Mapping[str, Any]): kwargs for vector store (currently only used if store_type == pinecone)
    """
    if store_type == "directory":
        directory = get_persist_dir(namespace_key)
        return __load_storage_context(persist_dir=directory)
    elif store_type == "pinecone":
        # namespace must be filtered at query time
        pinecone_index = get_vector_db(DEFAULT_PINECONE_INDEX)
        vector_store = PineconeVectorStore(pinecone_index, **vector_store_kwargs)
        return __load_storage_context(vector_store=vector_store)

    raise Exception(f"Unknown store type {store_type}")


def __get_llm(model_name: Optional[LlmModelType]):
    """
    Get llm based on model_name

    Args:
        model_name (LlmModelType): model to use for llm
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


def get_service_context(
    model_name: Optional[LlmModelType] = DEFAULT_MODEL_NAME, **kwargs
) -> ServiceContext:
    """
    Get default service context for llllamama index

    Args:
        model_name (Literal["ChatGPT", "VertexAI"]): model to use for llm
        **kwargs: additional kwargs to pass to ServiceContext.from_defaults
    """
    prompt_helper = PromptHelper(
        context_window=1024, num_output=256, chunk_overlap_ratio=0.1
    )

    llm = __get_llm(model_name)
    llm_predictor = LLMPredictor(llm=llm)

    service_context = ServiceContext.from_defaults(
        **kwargs, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )
    return service_context


ContextArgs = NamedTuple(
    "ContextArgs",
    [("model_name", Optional[LlmModelType]), ("storage_args", dict[str, Any])],
)

DEFAULT_CONTEXT_ARGS = ContextArgs(model_name=DEFAULT_MODEL_NAME, storage_args={})
