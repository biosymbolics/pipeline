"""
Functions specific to keyword table indices
"""
from typing import cast
from llama_index import GPTKeywordTableIndex, Response

from .general import get_or_create_index


def create_and_query_keyword_index(
    query: str, namespace: str, index_key: str, documents: list[str]
) -> str:
    """
    Creates the keyword table index if nx, and queries

    Args:
        query (str): natural language query
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_keyword_index(namespace, index_key, documents)
    response = index.as_query_engine().query(query)
    if not isinstance(response, Response) or not response.response:
        raise Exception("Could not parse response")
    return response.response


def get_keyword_index(
    namespace: str, index_id: str, documents: list[str]
) -> GPTKeywordTableIndex:
    """
    Creates or returns the vectore store index if nx, and queries

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_or_create_index(
        namespace,
        index_id,
        documents,
        index_impl=GPTKeywordTableIndex,  # type: ignore
    )
    return index
