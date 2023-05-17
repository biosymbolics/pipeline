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


def load_index(namespace: str, index_key: str):
    """
    Load persisted index
    """
    try:
        storage_context = StorageContext.from_defaults(persist_dir=namespace)
        index = load_index_from_storage(storage_context, index_id=index_key)
        return index
    except Exception as ex:
        logging.info("Could not load index from disk, perhaps nx? %s", ex)
        return None


def get_or_create_index(namespace: str, index_key: str, url: str):
    """
    Create singular llama index from supplied document url
    Skips creation if it already exists
    """
    index = load_index(namespace, index_key)
    if index:
        return index

    logging.info("Creating index %s/%s", namespace, index_key)
    try:
        documents = SimpleWebPageReader(html_to_text=True).load_data([url])
        index = GPTListIndex.from_documents(documents)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)


def create_and_query_index(query: str, namespace: str, index_key: str, url: str) -> str:
    """
    Creates the index if nx, and queries
    """
    get_or_create_index(namespace, index_key, url)
    return query_index(query, namespace, index_key)


def create_indicies(namespace: str, url_map: dict[str, str]):
    """
    Create indicies out of the supplied map (key: url)
    e.g. namespace: pfe-10q, url_map: {"2020-01-01": "https://sec.gov/pfizer-10q.htm"}
    """
    for key in url_map.keys():
        get_or_create_index(namespace, key, url_map[key])


def get_query_engine(namespace: str, index_key: str):
    """
    Get query engine for a given namespace/index
    """
    try:
        index = load_index(namespace, index_key)
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
