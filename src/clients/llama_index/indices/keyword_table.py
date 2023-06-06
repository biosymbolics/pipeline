"""
Functions specific to keyword table indices
"""
from llama_index import GPTKeywordTableIndex, Response

from .general import get_or_create_index


def create_and_query_keyword_index(
    query: str, namespace: str, index_id: str, documents: list[str]
) -> str:
    """
    Gets keyword table index and queries

    Args:
        query (str): natural language query
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_keyword_index(namespace, index_id, documents)
    response = index.as_query_engine().query(query)
    if not isinstance(response, Response) or not response.response:
        raise Exception("Could not parse response")
    return response.response


def get_keyword_index(
    namespace: str, index_id: str, documents: list[str]
) -> GPTKeywordTableIndex:
    """
    Creates or returns the vector store index; queries

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
