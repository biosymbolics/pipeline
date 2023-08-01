"""
Client for llama indexes
"""
import logging
import traceback
from typing import Any, Mapping, Optional, Type, TypeVar, Union
from llama_index import Document, Response, load_index_from_storage

from clients.llama_index.context import (
    get_service_context,
    get_storage_context,
    StorageArgs,
)
from clients.llama_index.formatting import format_documents
from constants.core import DEFAULT_MODEL_NAME
from typings.indices import LlmIndex, LlmModelType, Prompt, RefinePrompt

from ..types import GetDocId, GetDocMetadata

IndexImpl = TypeVar("IndexImpl", bound=LlmIndex)


def load_index(
    index_name: str,
    index_impl: Type[IndexImpl],
    model_name: LlmModelType = DEFAULT_MODEL_NAME,
    storage_args: StorageArgs = {},
    index_args: Mapping[str, Any] = {},
) -> LlmIndex:
    """
    Load persisted index. Creates new index if not found.

    Args:
        index_name (str): name of the index
        context_args (ContextArgs): context args for loading index
        index_impl (Type[IndexImpl]): index implementation, defaults to VectorStoreIndex
        index_args (Mapping[str, Any]): args to pass to the index implementation
    """
    logging.info(
        "Loading context for index %s (%s, %s)",
        index_name,
        model_name,
        storage_args,
        index_impl,
    )
    storage_context = get_storage_context(index_name, **storage_args)
    service_context = get_service_context(model_name=model_name)

    try:
        index = load_index_from_storage(storage_context)
    except Exception as e:
        logging.info("Cannot load index; creating")
        index = index_impl(
            [],
            storage_context=storage_context,
            service_context=service_context,
            **index_args,
        )

    logging.info("Returning index %s", index_name)

    return index


def query_index(
    index: LlmIndex,
    query: str,
    prompt_template: Optional[Prompt] = None,
    refine_prompt: Optional[RefinePrompt] = None,
    **kwargs,
) -> str:
    """
    Queries a given index

    Args:
        index (LlmIndex): llama index
        query (str): natural language query
        prompt_template (QuestionAnswerPrompt): prompt template to use for query (optional; will be defaulted)
        refine_prompt (RefinePrompt): prompt to use for refine (optional)
        **kwargs: additional args to pass to the query engine
    """
    if prompt_template and refine_prompt:
        query_engine = index.as_query_engine(
            **kwargs,
            text_qa_template=prompt_template,
            refine_template=refine_prompt,
        )
    else:
        query_engine = index.as_query_engine(**kwargs)

    response = query_engine.query(query)

    if not isinstance(response, Response) or not response.response:
        logging.error("Could not parse response: %s", response)
        raise Exception("Could not parse response")

    return response.response


def upsert_index(
    index_name: str,
    documents: Union[list[str], list[Document]],
    index_impl: Type[IndexImpl],
    index_args: Mapping[str, Any] = {},
    model_name: LlmModelType = DEFAULT_MODEL_NAME,
    storage_args: StorageArgs = {},
    get_doc_metadata: Optional[GetDocMetadata] = None,
    get_doc_id: Optional[GetDocId] = None,
) -> LlmIndex:
    """
    Create or add to an index from supplied document url

    Args:
        index_name (str): name of the index
        documents (Document): list of strings or docs
        index_impl (IndexImpl): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
        model_name (LlmModelType): name of the model to use
        storage_args (StorageArgs): storage args for loading index
        get_doc_metadata (GetDocMetadata): function to get extra info to put on docs (metadata)
        get_doc_id (GetDocId): function to get doc id from doc
    """
    logging.info("Adding docs to index %s", index_name)

    index = load_index(index_name, index_impl, model_name, storage_args, index_args)

    try:
        ll_docs = format_documents(documents, get_doc_metadata, get_doc_id)
        index.refresh_ref_docs(ll_docs)  # adds if nx, updates if hash is different
    except Exception as ex:
        logging.error("Error upserting index %s: %s", index_name, ex)
        traceback.print_exc()
        raise ex

    return index
