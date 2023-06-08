"""
Client for llama indexes
"""
import logging
import traceback
from typing import Any, Mapping, Optional, TypeVar, Union
from llama_index import Document, Response

from clients.llama_index.context import (
    get_service_context,
    get_storage_context,
    ContextArgs,
    DEFAULT_CONTEXT_ARGS,
)
from clients.llama_index.formatting import format_documents
from clients.llama_index.persistence import load_index
from local_types.indices import LlmIndex, Prompt, RefinePrompt

from ..types import GetDocMetadata

IndexImpl = TypeVar("IndexImpl", bound=LlmIndex)


def get_index(
    index_name: str,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> LlmIndex:
    """
    Get llama index from the namespace/index_id

    Args:
        index_name (str): name of the index
        context_args (ContextArgs): context args for loading index
    """
    return load_index(index_name, context_args)


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
        logging.error("Could not parse response: %s", response)
        raise Exception("Could not parse response")

    return response.response


def upsert_index(
    index_name: str,
    documents: Union[list[str], list[Document]],
    index_impl: IndexImpl,
    index_args: Mapping[str, Any] = {},
    get_doc_metadata: Optional[GetDocMetadata] = None,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> IndexImpl:
    """
    Create an index from supplied document url

    Note: this is a bit of a misnomer for pinecone, for which we're just adding new documents to an existing index

    Args:
        index_name (str): name of the index
        documents (Document): list of strings or docs
        index_impl (IndexImpl): the llama index type to use
        index_args (dict): args to pass to the LlmIndex obj
        context_args (ContextArgs): context args for loading index
        get_doc_metadata (GetDocMetadata): function to get extra info to put on docs (metadata)
    """
    logging.info("Creating index %s", index_name)

    try:
        ll_docs = format_documents(documents, get_doc_metadata)
        index = index_impl.from_documents(
            ll_docs,
            service_context=get_service_context(model_name=context_args.model_name),
            storage_context=get_storage_context(
                index_name, **context_args.storage_args
            ),
            *index_args,
        )
        return index
    except Exception as ex:
        logging.error("Error creating index %s: %s", index_name, ex)
        traceback.print_exc()
        raise ex
