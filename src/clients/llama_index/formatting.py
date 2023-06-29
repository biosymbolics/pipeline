"""
Utils for formatting related to llama index
"""
from typing import Any, Optional, TypeGuard, Union, cast
from llama_index import Document
from pydash import compact

from .types import GetDocId, GetDocMetadata


def __is_string_doc_list(documents: list[Any]) -> TypeGuard[list[str]]:
    """
    Check if documents are strings or llama_index Documents

    Args:
        documents (list[Any]): list of documents
    """
    return isinstance(documents[0], str)


def format_documents(
    documents: Union[list[str], list[Document]],
    get_extra_info: Optional[GetDocMetadata] = None,
    get_doc_id: Optional[GetDocId] = None,
) -> list[Document]:
    """
    Format documents to list of llama_index Documents, optionally adding metadata (extra_info) and doc_id

    Args:
        documents (list[str] or list[Document]): list of documents
        get_extra_info (GetExtraInfo): function to get extra info for each document
        get_doc_id (GetDocId): function to get doc id for each document
    """
    if __is_string_doc_list(documents):
        docs = list(map(lambda doc: Document(text=doc), documents))
    else:
        docs = cast(list[Document], documents)

    docs = compact(docs)  # why?

    if get_extra_info:
        for doc in docs:
            doc.metadata = get_extra_info(doc)

    if get_doc_id:
        for doc in docs:
            doc.id_ = get_doc_id(doc)

    return docs
