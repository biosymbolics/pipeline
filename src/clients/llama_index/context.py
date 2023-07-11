"""
Functions around llama index context
"""
from abc import abstractmethod
import os
import sys
from typing import Any, Callable, Literal, NamedTuple, Optional, cast
from llama_index import (
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    StorageContext,
)
from llama_index.storage.docstore import MongoDocumentStore
from llama_index.storage.index_store import MongoIndexStore
from llama_index.vector_stores import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.llms import Anthropic, VertexAI
from llama_index.vector_stores import ChromaVectorStore
import chromadb
from chromadb.config import Settings
import logging

from constants.core import DEFAULT_MODEL_NAME
from clients.stores.pinecone import get_vector_store
from typings.indices import LlmModelType

ContextArgs = NamedTuple(
    "ContextArgs",
    [("model_name", Optional[LlmModelType]), ("storage_args", dict[str, Any])],
)

logging.basicConfig(level=logging.DEBUG)

DEFAULT_CONTEXT_ARGS = ContextArgs(model_name=DEFAULT_MODEL_NAME, storage_args={})
MONGO_URI = os.environ["MONGO_URI"]


def get_storage_context(
    index_name: str,
    storage_type: Literal["pinecone", "mongodb"] = "pinecone",
    **kwargs,
) -> StorageContext:
    """
    Get vector store storage context

    Args:
        index_name (str): name of the index
        kwargs (Mapping[str, Any]): kwargs for vector store
    """
    logging.info("Loading storage context for %s", index_name)

    if storage_type == "pinecone":
        logging.info("Loading pinecone vector store context")
        pinecone_index = get_vector_store(index_name)
        vector_store = PineconeVectorStore(pinecone_index, **kwargs)
        context = StorageContext.from_defaults(vector_store=vector_store)
        return context

    elif storage_type == "mongodb":
        logging.info(
            "Loading mongodb doc and index store context, chromadb vector store"
        )
        chroma_client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./storage/vector_storage/chromadb/",
            )
        )

        chroma_collection = chroma_client.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index_store = MongoIndexStore.from_uri(
            uri=MONGO_URI, db_name="db_docstore", namespace="index_store"
        )

        return StorageContext.from_defaults(
            # **kwargs
            docstore=MongoDocumentStore.from_uri(
                uri=MONGO_URI, db_name="db_docstore", namespace="docstore"
            ),
            index_store=index_store,
            vector_store=vector_store,
        )

    raise Exception(f"Unknown storage type {storage_type}")


def get_service_context(
    model_name: Optional[LlmModelType] = DEFAULT_MODEL_NAME, **kwargs
) -> ServiceContext:
    """
    Get default service context for llllamama index

    Args:
        model_name (Literal["ChatGPT", "VertexAI"]): model to use for llm
        **kwargs: additional kwargs to pass to ServiceContext.from_defaults
    """

    def __get_llm(model_name: Optional[LlmModelType]):
        """
        Get llm based on model_name
        """
        if model_name == "ChatGPT":
            return ChatOpenAI(
                model="gpt-3.5-turbo-16k",
                max_tokens=10000,
                client="chat",
                temperature=0.1,
            )
        if model_name == "VertexAI":
            return VertexAI(model_name="text-bison", temperature=0.1)
        if model_name == "Anthropic":
            # untested, but used in https://colab.research.google.com/drive/1uuqvPI2_WNFMd7g-ahFoioSHV7ExB2GR?usp=sharing
            # benefit is massive input token limit. use with GPTListIndex?
            return Anthropic(model="claude-v1.3-100k", temperature=0.1)

        raise Exception(f"Unknown model {model_name}")

    llm = __get_llm(model_name)
    llm_predictor = LLMPredictor(llm=llm)
    prompt_helper = PromptHelper.from_llm_metadata(llm_predictor.metadata)

    logging.info("Prompt helper: %s", prompt_helper.__dict__.items())

    service_context = ServiceContext.from_defaults(
        **kwargs,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
    )

    service_context.llama_logger.get_logs()
    return service_context
