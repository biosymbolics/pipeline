"""
Client for llama indexes
"""
import os
import logging
from typing import Type
from llama_index import ComposableGraph
from llama_index.indices.base import BaseGPTIndex as LlmIndex
from llama_index.indices.query.base import BaseQueryEngine

from clients.llama_index.persistence import (
    load_index,
    load_indices,
)

from .context import get_service_context, get_storage_context

API_KEY = os.environ["OPENAI_API_KEY"]


def compose_graph(namespace: str, index_type: Type[LlmIndex]) -> ComposableGraph:
    """
    Composes graph for all indices in namespace

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
    """
    indices = load_indices(namespace)
    index_summary = [
        index.as_query_engine().query("Summary this document in 100 words").response
        for index in indices
    ]
    graph = ComposableGraph.from_indices(
        index_type,
        children_indices=indices,
        index_summaries=index_summary,
        service_context=get_service_context(),
        storage_context=get_storage_context(namespace),
    )
    return graph


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
