"""
Utility for llama indexes
"""
import os
import logging
from llama_index import GPTListIndex, SimpleWebPageReader

API_KEY = os.environ["OPENAI_API_KEY"]

# redis? pinecone?
index_map: dict[str, dict[str, GPTListIndex]] = {}


def create_index(namespace: str, index_key: str, url: str):
    """
    Create singular llama index from supplied document url
    Skips creation if it already exists
    """
    if index_key in index_map.get(namespace, {}):
        logging.info("Not recreating index %s/%s", namespace, index_key)
        return

    logging.info("Creating index %s/%s", namespace, index_key)
    try:
        documents = SimpleWebPageReader(html_to_text=True).load_data([url])
        index = GPTListIndex.from_documents(documents)
        if namespace not in index_map:
            index_map[namespace] = {}
        index_map[namespace][index_key] = index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)


def create_and_query_index(query: str, namespace: str, index_key: str, url: str) -> str:
    """
    Creates the index if nx, and queries
    """
    create_index(namespace, index_key, url)
    return query_index(query, namespace, index_key)


def create_indicies(namespace: str, url_map: dict[str, str]):
    """
    Create indicies out of the supplied map (key: url)
    e.g. namespace: pfe-10q, url_map: {"2020-01-01": "https://sec.gov/pfizer-10q.htm"}
    """
    for key in url_map.keys():
        create_index(namespace, key, url_map[key])


def get_query_engine(namespace: str, index_key: str):
    """
    Get query engine for a given namespace/index
    """
    try:
        query_engine = index_map[namespace][index_key].as_query_engine()
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
