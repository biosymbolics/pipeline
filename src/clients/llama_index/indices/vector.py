"""
Functions specific to vector store indices
"""
from llama_index import GPTVectorStoreIndex

from clients.llama_index.types import NamespaceKey

from .general import get_or_create_index, get_index, query_index


def query_vector_index(query: str, namespace_key: NamespaceKey, index_id: str) -> str:
    """
    Queries a given vector store index

    Args:
        query (str): natural language query
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = get_index(namespace_key, index_id)
    answer = query_index(index, query)
    return answer


def create_and_query_vector_index(
    query: str, namespace_key: NamespaceKey, index_id: str, documents: list[str]
) -> str:
    """
    Gets the vector store index and queries

    Args:
        query (str): natural language query
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_vector_index(namespace_key, index_id, documents)
    answer = query_index(index, query)
    return answer


def get_vector_index(
    namespace_key: NamespaceKey, index_id: str, documents: list[str]
) -> GPTVectorStoreIndex:
    """
    Creates or returns the vectore store index if nx, and queries

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    return get_or_create_index(
        namespace_key,
        index_id,
        documents,
        index_impl=GPTVectorStoreIndex,  # type: ignore
    )
