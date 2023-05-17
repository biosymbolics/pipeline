"""
Utility for llama indexes
"""
import os
import logging
from llama_index import (
    GPTListIndex,
    SimpleWebPageReader,
    StorageContext,
    load_index_from_storage,
)

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_STORAGE_DIR = "./storage"


def __get_persist_dir(namespace: str) -> str:
    return f"{BASE_STORAGE_DIR}/{namespace}"


def __persist_index(index: GPTListIndex, namespace: str):
    """
    Persist llama index
    """
    try:
        directory = __get_persist_dir(namespace)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)


def __load_index(namespace: str, index_key: str) -> GPTListIndex:
    """
    Load persisted index
    """
    try:
        directory = __get_persist_dir(namespace)
        storage_context = StorageContext.from_defaults(persist_dir=directory)
        index = load_index_from_storage(storage_context, index_id=index_key)
        return index
    except Exception as ex:
        logging.info("Could not load index from disk. nx? %s", ex)
        return None


def get_or_create_index(namespace: str, index_key: str, url: str) -> GPTListIndex:
    """
    Create singular llama index from supplied document url
    Skips creation if it already exists
    """
    index = __load_index(namespace, index_key)
    if index:
        return index

    logging.info("Creating index %s/%s", namespace, index_key)
    try:
        documents = SimpleWebPageReader(html_to_text=True).load_data([url])
        index = GPTListIndex.from_documents(documents)
        __persist_index(index, namespace)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)
        return None


def create_and_query_index(query: str, namespace: str, index_key: str, url: str) -> str:
    """
    Creates the index if nx, and queries
    """
    index = get_or_create_index(namespace, index_key, url)
    return index.as_query_engine().query(query)


def get_query_engine(namespace: str, index_key: str):
    """
    Get query engine for a given namespace/index
    """
    try:
        index = __load_index(namespace, index_key)
        if not index:
            logging.error("Index %s/%s not found", namespace, index_key)
            raise Exception("Index not found")
        query_engine = index.as_query_engine()
        return query_engine
    except Exception as ex:
        logging.error("Error generating query engine: %s", ex)
        raise ex


def query_index(query: str, namespace: str, index_key: str) -> str:
    """
    Query the specified index
    """
    try:
        query_engine = get_query_engine(namespace, index_key)
        response = query_engine.query(query)
        return response
    except Exception as ex:
        logging.error("Error querying index: %s", ex)
        return ""
