"""
Functions specific to composed indices
"""
import logging
from typing import Type, cast
from llama_index import ComposableGraph, GPTVectorStoreIndex
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from clients.llama_index.constants import DEFAULT_MODEL_NAME
from clients.llama_index.context import get_service_context, get_storage_context
from clients.llama_index.persistence import load_indices
from types.indices import LlmModel, NamespaceKey
from .general import query_index


def __compose_graph(
    namespace_key: NamespaceKey,
    index_type: Type[LlmIndex],
    model_name: LlmModel = DEFAULT_MODEL_NAME,
) -> ComposableGraph:
    """
    Composes graph for all indices in namespace

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
    """
    indices = load_indices(namespace_key)
    index_summary = [
        index.as_query_engine().query("Summary this document in 100 words").__str__()
        for index in indices
    ]
    service_context = get_service_context(model_name)
    graph = ComposableGraph.from_indices(
        index_type,
        children_indices=indices,
        index_summaries=index_summary,
        service_context=service_context,
        storage_context=get_storage_context(namespace_key),
    )
    return graph


def get_composed_index(
    namespace_key: NamespaceKey, model_name: LlmModel = DEFAULT_MODEL_NAME
) -> GPTVectorStoreIndex:
    """
    Get llama index from the namespace/index_id

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        model_name (LlmModel): model name to use for index (optional)
    """
    index = __compose_graph(namespace_key, GPTVectorStoreIndex, model_name)
    return cast(GPTVectorStoreIndex, index)


def query_composed_index(
    query: str, namespace_key: NamespaceKey, model_name: LlmModel = DEFAULT_MODEL_NAME
) -> str:
    """
    Forms and queries a composed index

    Args:
        query (str): natural language query
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        model_name (LlmModel): model name to use for index (optional)
    """
    index = get_composed_index(namespace_key, model_name)
    return query_index(index, query)
