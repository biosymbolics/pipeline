"""
Client for llama indexes
"""
import logging
from typing import Optional, TypeVar, cast
from llama_index import Document, QuestionAnswerPrompt, Response, RefinePrompt
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from clients.llama_index.context import get_service_context, get_storage_context
from clients.llama_index.persistence import load_index, persist_index


IndexImpl = TypeVar("IndexImpl", bound=LlmIndex)


def get_index(
    namespace: str,
    index_id: str,
) -> LlmIndex:
    """
    Get llama index from the namespace/index_id

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = load_index(namespace, index_id)
    if index:
        return index
    raise Exception("Index not found")


def query_index(
    index: LlmIndex,
    query: str,
    prompt: Optional[QuestionAnswerPrompt] = None,
    refine_prompt: Optional[RefinePrompt] = None,
) -> str:
    """
    Queries a given index

    Args:
        index (LlmIndex): llama index
        query (str): natural language query
    """
    query_engine = index.as_query_engine(
        text_qa_template=prompt,
        refine_template=refine_prompt,
    )

    response = query_engine.query(query)
    if not isinstance(response, Response) or not response.response:
        raise Exception("Could not parse response")
    return response.response


def get_or_create_index(
    namespace: str,
    index_id: str,
    documents: list[str],
    index_impl: IndexImpl,
    index_args: Optional[dict] = None,
) -> IndexImpl:
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
        return cast(IndexImpl, index)

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
