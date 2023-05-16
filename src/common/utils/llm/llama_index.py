"""
Utility for llama indexes
"""
import os
import logging
from llama_index import GPTListIndex, SimpleWebPageReader

API_KEY = os.environ["OPENAI_API_KEY"]

# redis? pinecone?
index_map: dict[str, dict[str, GPTListIndex]] = {}


def create_indicies(namespace: str, url_map: dict[str, str]):
    """
    Create indicies out of the supplied map (key: url)
    e.g. namespace: pfe-10q, url_map: {"2020-01-01": "https://sec.gov/pfizer-10q.htm"}
    """
    for key in url_map.keys():
        documents = SimpleWebPageReader(html_to_text=True).load_data([url_map[key]])
        index = GPTListIndex.from_documents(documents)
        index_map[namespace][key] = index


def get_query_engine(namespace: str, index_key: str):
    """
    Get query engine for a given namespace/index
    """
    try:
        query_engine = index_map[namespace][index_key].as_query_engine()
        return query_engine
    except Exception as ex:
        logging.error("Error generating query engine: %s", ex)
        return None


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
