"""
Utility for llama indexes
"""
import os
import logging
from llama_index import (
    Document,
    GPTListIndex,
    StorageContext,
    load_index_from_storage,
)


API_KEY = os.environ["OPENAI_API_KEY"]
BASE_STORAGE_DIR = "./storage"
SEC_DOCS_DIR = "./sec_docs"


def __get_persist_dir(namespace: str) -> str:
    return f"{BASE_STORAGE_DIR}/{namespace}"


def __persist_index(index: GPTListIndex, namespace: str, index_id: str):
    """
    Persist llama index
    """
    try:
        directory = __get_persist_dir(namespace)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)


def __load_index(namespace: str, index_id: str) -> GPTListIndex:
    """
    Load persisted index
    """
    try:
        directory = __get_persist_dir(namespace)
        storage_context = StorageContext.from_defaults(persist_dir=directory)
        index = load_index_from_storage(storage_context, index_id=index_id)

        logging.info("Returning index %s/%s from disk", namespace, index_id)
        return index
    except Exception as ex:
        logging.info("Failed to load %s/%s from disk: %s", namespace, index_id, ex)
        return None


def get_or_create_index(
    namespace: str, index_id: str, documents: list[str]
) -> GPTListIndex:
    """
    Create singular llama index from supplied document url
    Skips creation if it already exists
    """

    index = __load_index(namespace, index_id)
    if index:
        return index

    logging.info("Creating index %s/%s", namespace, index_id)
    try:
        ll_docs = list(map(Document, documents))
        index = GPTListIndex.from_documents(ll_docs)
        __persist_index(index, namespace, index_id)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)
        raise ex


def create_and_query_index(
    query: str, namespace: str, index_key: str, documents: list[str]
) -> str:
    """
    Creates the index if nx, and queries
    """
    index = get_or_create_index(namespace, index_key, documents)
    response = index.as_query_engine().query(query)
    return response.response


def get_query_engine(namespace: str, index_id: str):
    """
    Get query engine for a given namespace/index
    """
    try:
        index = __load_index(namespace, index_id)
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
    """
    try:
        query_engine = get_query_engine(namespace, index_id)
        response = query_engine.query(query)
        return response
    except Exception as ex:
        logging.error("Error querying index: %s", ex)
        return ""
