"""
Client for llama indexes
"""
import logging
from typing import Optional
from llama_index import Document
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from clients.llama_index.context import get_service_context, get_storage_context
from clients.llama_index.persistence import load_index, persist_index


def get_or_create_index(
    namespace: str,
    index_id: str,
    documents: list[str],
    index_impl: LlmIndex,
    index_args: Optional[dict] = None,
) -> LlmIndex:
    """
    Create llama index from supplied document url
    Skips creation if it already exists

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
        LlmIndex (LlmIndex): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
    """

    index = load_index(namespace, index_id)
    if index:
        return index

    logging.info("Creating index %s/%s", namespace, index_id)
    service_context = get_service_context()
    try:
        ll_docs = list(map(Document, documents))
        index = index_impl.from_documents(
            ll_docs,
            service_context=service_context,
            storage_context=get_storage_context(namespace),
            *(index_args or {})
        )
        persist_index(index, namespace, index_id)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)
        raise ex
