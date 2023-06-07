"""
Client for llama indexes
"""
import logging
from typing import Any, Mapping, Optional, TypeVar, cast
from llama_index import Document, Response
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from constants.core import DEFAULT_MODEL_NAME
from clients.llama_index.context import get_service_context, get_storage_context
from clients.llama_index.persistence import maybe_load_index, persist_index
from types.indices import LlmModelType, NamespaceKey, Prompt, RefinePrompt

IndexImpl = TypeVar("IndexImpl", bound=LlmIndex)


def get_index(
    namespace_key: NamespaceKey,
    index_id: str,
) -> LlmIndex:
    """
    Get llama index from the namespace/index_id

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. ("BIBB", "SEC", "10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = maybe_load_index(namespace_key, index_id)
    if index:
        return index
    raise Exception("Index not found")


def query_index(
    index: LlmIndex,
    query: str,
    prompt: Optional[Prompt] = None,
    refine_prompt: Optional[RefinePrompt] = None,
) -> str:
    """
    Queries a given index

    Args:
        index (LlmIndex): llama index
        query (str): natural language query
        prompt (QuestionAnswerPrompt): prompt to use for query (optional)
        refine_prompt (RefinePrompt): prompt to use for refine (optional)
    """
    if prompt and refine_prompt:
        query_engine = index.as_query_engine(
            text_qa_template=prompt,
            refine_template=refine_prompt,
        )
    else:
        query_engine = index.as_query_engine()

    response = query_engine.query(query)

    if not isinstance(response, Response) or not response.response:
        raise Exception("Could not parse response")

    return response.response


def create_index(
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
    index_impl: IndexImpl,
    index_args: Optional[dict] = {},
    model_name: Optional[LlmModelType] = DEFAULT_MODEL_NAME,
    storage_context_args: Mapping[str, Any] = {},
) -> IndexImpl:
    """
    Create llama index from supplied document url

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. ("BIBB", "SEC", "10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
        index_impl (IndexImpl): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
        model_name (LlmModelType): llm model to use
    """
    logging.info("Creating index %s/%s", namespace_key, index_id)
    try:
        ll_docs = list(map(Document, documents))
        index = index_impl.from_documents(
            ll_docs,
            service_context=get_service_context(model_name),
            storage_context=get_storage_context(namespace_key, **storage_context_args),
            *index_args,
        )
        persist_index(index, namespace_key, index_id)
        return index
    except Exception as ex:
        logging.error("Error creating index: %s", ex)
        raise ex


def get_or_create_index(
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
    index_impl: IndexImpl,
    index_args: Optional[dict] = None,
    model_name: Optional[LlmModelType] = DEFAULT_MODEL_NAME,
) -> IndexImpl:
    """
    If nx, create llama index from supplied documents. Otherwise return existing index.

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. ("BIBB", "SEC", "10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
        index_impl (IndexImpl): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
        model_name (LlmModelType): llm model to use
    """
    index = maybe_load_index(namespace_key, index_id)
    if index:
        return cast(IndexImpl, index)

    return create_index(
        namespace_key, index_id, documents, index_impl, index_args, model_name
    )
