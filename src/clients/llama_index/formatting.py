"""
Utils for formatting related to llama index
"""
from typing import Any, Optional, TypeGuard, Union, cast
from llama_index import Document
from pydash import compact

from .types import GetDocMetadata


def __is_string_doc_list(documents: list[Any]) -> TypeGuard[list[str]]:
    """
    Check if documents are strings or llama_index Documents
    """
    return isinstance(documents[0], str)


def format_documents(
    documents: Union[list[str], list[Document]],
    get_extra_info: Optional[GetDocMetadata] = None,
) -> list[Document]:
    """
    Format documents to list of llama_index Documents

    Args:
        documents (list[str] or list[Document]): list of documents
        get_extra_info (GetExtraInfo): function to get extra info for each document
    """
    if __is_string_doc_list(documents):
        docs = list(map(Document, documents))
    else:
        docs = cast(list[Document], documents)

    docs = compact(docs)

    if get_extra_info:
        for doc in docs:
            doc.extra_info = get_extra_info(doc)

    return docs
