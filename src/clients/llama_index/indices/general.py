"""
Client for llama indexes
"""
import logging
from typing import Optional, TypeVar, Union, cast
from llama_index import Document, Response
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from constants.core import DEFAULT_MODEL_NAME
from clients.llama_index.context import get_service_context, get_storage_context
from clients.llama_index.formatting import format_documents
from clients.llama_index.persistence import maybe_load_index, persist_index
from types.indices import LlmModelType, NamespaceKey, Prompt, RefinePrompt

from ..types import GetDocMetadata

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
    **kwargs,
) -> str:
    """
    Queries a given index

    Args:
        index (LlmIndex): llama index
        query (str): natural language query
        prompt (QuestionAnswerPrompt): prompt to use for query (optional)
        refine_prompt (RefinePrompt): prompt to use for refine (optional)
        **kwargs: additional args to pass to the query engine

    TODO: filters=MetadataFilters(filters=[
        ExactMatchFilter(key='name', value='paul graham'),
    ]),
    """
    if prompt and refine_prompt:
        query_engine = index.as_query_engine(
            **kwargs,
            text_qa_template=prompt,
            refine_template=refine_prompt,
        )
    else:
        query_engine = index.as_query_engine(**kwargs)

    response = query_engine.query(query)

    if not isinstance(response, Response) or not response.response:
        raise Exception("Could not parse response")

    return response.response


def create_index(
    namespace_key: NamespaceKey,
    index_id: str,
    documents: Union[list[str], list[Document]],
    index_impl: IndexImpl,
    index_args: Optional[dict] = {},
    model_name: Optional[LlmModelType] = DEFAULT_MODEL_NAME,
    get_doc_metadata: Optional[GetDocMetadata] = None,
) -> IndexImpl:
    """
    Create llama index from supplied document url

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. ("BIBB", "SEC", "10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of strings or docs
        index_impl (IndexImpl): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
        model_name (LlmModelType): llm model to use
        get_doc_metadata (GetDocMetadata): function to get extra info to put on docs (metadata)
    """
    logging.info("Creating index %s/%s", namespace_key, index_id)
    try:
        ll_docs = format_documents(documents, get_doc_metadata)
        index = index_impl.from_documents(
            ll_docs,
            service_context=get_service_context(model_name),
            storage_context=get_storage_context(namespace_key),
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
