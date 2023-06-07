"""
Functions specific to composed indices
"""
import logging
from typing import Type, cast
from llama_index import ComposableGraph, GPTVectorStoreIndex

from clients.llama_index.context import (
    get_service_context,
    get_storage_context,
    DEFAULT_CONTEXT_ARGS,
    ContextArgs,
)
from clients.llama_index.persistence import load_indices
from local_types.indices import LlmIndex, NamespaceKey
from .general import query_index


def __compose_graph(
    namespace_key: NamespaceKey,
    index_type: Type[LlmIndex],
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> ComposableGraph:
    """
    Composes graph for all indices in namespace

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_type (Type[LlmIndex]): index type to compose
        context_args (ContextArgs): context args for loading index
    """
    indices = load_indices(namespace_key)
    index_summary = [
        index.as_query_engine().query("Summary this document in 100 words").__str__()
        for index in indices
    ]
    service_context = get_service_context(context_args.model_name)
    graph = ComposableGraph.from_indices(
        index_type,
        children_indices=indices,
        index_summaries=index_summary,
        service_context=service_context,
        storage_context=get_storage_context(namespace_key, **context_args.storage_args),
    )
    return graph


def get_composed_index(
    namespace_key: NamespaceKey,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> GPTVectorStoreIndex:
    """
    Get llama index from the namespace/index_id

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        context_args (ContextArgs): context args for loading index
    """
    index = __compose_graph(namespace_key, GPTVectorStoreIndex, context_args)
    return cast(GPTVectorStoreIndex, index)


def query_composed_index(
    query: str,
    namespace_key: NamespaceKey,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> str:
    """
    Forms and queries a composed index

    Args:
        query (str): natural language query
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        context_args (ContextArgs): context args for loading index
    """
    index = get_composed_index(namespace_key, context_args)
    return query_index(index, query)
