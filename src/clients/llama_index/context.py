"""
Functions around llama index context
"""
import os
from typing import Any, Literal, Optional, TypedDict
from typing_extensions import NotRequired
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
import logging

from constants.core import DEFAULT_MODEL_NAME
from clients.stores.pinecone import get_vector_store
from typings.indices import LlmModelType

StorageArgs = dict[str, Any]


class ModelInfo(TypedDict):
    max_tokens: NotRequired[int]
    model: str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.environ.get("MONGO_URI")


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
    logger.info("Loading storage context for %s", index_name)

    if MONGO_URI is None:
        raise ValueError("MONGO_URI not set")

    if storage_type == "pinecone":
        logger.info("Loading pinecone vector store context")
        pinecone_index = get_vector_store(index_name)
        vector_store = PineconeVectorStore(pinecone_index, **kwargs)
        return StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(
                uri=MONGO_URI, db_name="db_docstore", namespace="docstore"
            ),
            index_store=MongoIndexStore.from_uri(
                uri=MONGO_URI, db_name="db_docstore", namespace="index_store"
            ),
            vector_store=vector_store,
        )

    elif storage_type == "mongodb":
        logger.info(
            "Loading mongodb doc and index store context, chromadb vector store"
        )
        # lazy import (bloat!)
        import chromadb

        chroma_client = chromadb.PersistentClient(
            path="./storage/vector_storage/chromadb/",
        )

        chroma_collection = chroma_client.get_or_create_collection(index_name)

        logger.info(
            "Loaded chroma collection; contains %s docs", chroma_collection.count()
        )

        return StorageContext.from_defaults(
            docstore=MongoDocumentStore.from_uri(
                uri=MONGO_URI, db_name="db_docstore", namespace="docstore"
            ),
            index_store=MongoIndexStore.from_uri(
                uri=MONGO_URI, db_name="db_docstore", namespace="index_store"
            ),
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection),
        )

    raise Exception(f"Unknown storage type {storage_type}")


OUTPUT_TOKENS = 6000
MODEL_TO_TOKENS: dict[LlmModelType, dict] = {
    "ChatGPT": {"model": "gpt-3.5-turbo-16k"},
    "GPT4": {"model": "gpt-4"},  # "gpt-4-32k"
    "VertexAI": {"model": "text-bison"},
    "Anthropic": {"max_tokens": 100000 - OUTPUT_TOKENS, "model": "claude-v1.3-100k"},
}


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
        if model_name in ["ChatGPT", "GPT4"]:
            return ChatOpenAI(
                **MODEL_TO_TOKENS[model_name],
                client="chat",
                temperature=0.1,
            )
        if model_name == "VertexAI":
            return VertexAI(**MODEL_TO_TOKENS[model_name], temperature=0.1)
        if model_name == "Anthropic":
            # untested, but used in https://colab.research.google.com/drive/1uuqvPI2_WNFMd7g-ahFoioSHV7ExB2GR?usp=sharing
            return Anthropic(model_name="claude-v1.3-100k", temperature=0.1)

        raise Exception(f"Unknown model {model_name}")

    llm = __get_llm(model_name)
    llm_predictor = LLMPredictor(llm=llm)
    prompt_helper = PromptHelper.from_llm_metadata(llm_predictor.metadata)

    logger.info("Prompt helper: %s", prompt_helper.__dict__.items())

    service_context = ServiceContext.from_defaults(
        **kwargs,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
    )

    service_context.llama_logger.get_logs()
    return service_context
