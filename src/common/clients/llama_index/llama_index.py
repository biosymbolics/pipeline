"""
Utility for llama indexes
"""
import os
import logging
from llama_index import (
    Document,
    load_index_from_storage,
)
from llama_index.indices.base import BaseGPTIndex
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
from llama_index.indices.query.base import BaseQueryEngine

from sources.sec.prompts import BIOMEDICAL_TRIPLET_EXTRACT_PROMPT

from .context import get_service_context, get_storage_context
from .utils import get_persist_dir

API_KEY = os.environ["OPENAI_API_KEY"]


def __persist_index(index: BaseGPTIndex, namespace: str, index_id: str):
    """
    Persist llama index

    Args:
        index (BaseGPTIndex): any generic GPT Index
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
    """
    try:
        directory = get_persist_dir(namespace)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)


def load_index(namespace: str, index_id: str) -> BaseGPTIndex:
    """
    Load persisted index

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
    """
    try:
        storage_context = get_storage_context(namespace)
        index = load_index_from_storage(
            storage_context,
            index_id=index_id,
            service_context=get_service_context(),
        )

        logging.info("Returning index %s/%s from disk", namespace, index_id)
        return index
    except Exception as ex:
        logging.info("Failed to load %s/%s from disk: %s", namespace, index_id, ex)
        print(ex)
        return None



def get_or_create_index(
    namespace: str, index_id: str, documents: list[str]
) -> GPTKnowledgeGraphIndex:
    """
    Create llama index from supplied document url
    Skips creation if it already exists
    
    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
        documents (Document): list of llama_index Documents
    """

    index = load_index(namespace, index_id)
    if index:
        return index

    logging.info("Creating index %s/%s", namespace, index_id)
    service_context = get_service_context()
    try:
        ll_docs = list(map(Document, documents))
        index = GPTKnowledgeGraphIndex.from_documents(
            ll_docs,
            service_context=service_context,
            storage_context=get_storage_context(namespace),
            kg_triple_extract_template=BIOMEDICAL_TRIPLET_EXTRACT_PROMPT,
            max_knowledge_triplets=50
        )
        __persist_index(index, namespace, index_id)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)
        raise ex


def create_and_query_index(query: str, namespace: str, index_key: str, documents: list[str]) -> str:
    """
    Creates the index if nx, and queries

    Args:
        query (str): natural language query
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
        documents (Document): list of llama_index Documents
    """
    index = get_or_create_index(namespace, index_key, documents)
    response = index.as_query_engine().query(query)
    return response.response


def get_query_engine(namespace: str, index_id: str) -> BaseQueryEngine:
    """
    Get query engine for a given namespace/index

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
    """
    try:
        index = load_index(namespace, index_id)
        if not index:
            logging.error("Index %s/%s not found", namespace, index_id)
            raise Exception("Index not found")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as ex:
        logging.error("Error generating query engine: %s", ex)
        raise ex


def query_index(query: str, namespace: str, index_id: str) -> str:
    """
    Query the specified index

    Args:
        query (str): natural language query
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1) 
    """
    try:
        query_engine = get_query_engine(namespace, index_id)
        response = query_engine.query(query)
        return response
    except Exception as ex:
        logging.error("Error querying index: %s", ex)
        return ""
