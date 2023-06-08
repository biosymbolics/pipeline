"""
SourceDocIndex
"""
from datetime import datetime
from typing import Optional
from llama_index import GPTVectorStoreIndex

from clients.llama_index import create_index, query_index
from clients.llama_index.context import ContextArgs, DEFAULT_CONTEXT_ARGS
from clients.llama_index.types import DocMetadata
from local_types.indices import LlmIndex, NamespaceKey, Prompt, RefinePrompt
from src.clients.vector_dbs.pinecone import get_metadata_filters

INDEX_NAME = "source_docs"


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs
    """

    def __init__(
        self,
        documents: Optional[list[str]] = None,
        source: Optional[NamespaceKey] = None,
        context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
        index_impl: LlmIndex = GPTVectorStoreIndex,  # type: ignore
        retrieval_date: datetime = datetime.now(),
    ):
        """
        initialize

        Args:
            context_args (ContextArgs, optional): context args. Defaults to DEFAULT_CONTEXT_ARGS.
            documents (list[str], optional): list of documents. Defaults to None. Can be loaded later with from_documents().
            index_impl (LlmIndex, optional): index implementation. Defaults to GPTKeywordTableIndex.
            retrieval_date (datetime, optional): retrieval date of source docs. Defaults to datetime.now().
        """
        self.context_args = context_args
        self.index = None  # TODO: load
        self.index_impl = index_impl
        self.retrieval_date = retrieval_date
        self.source = source

        if documents and source:
            self.from_documents(documents, source)
        elif documents or source:
            raise ValueError("Must provide both documents and source or neither.")

    def from_documents(self, documents: list[str], source: NamespaceKey):
        """
        Load docs into index with metadata

        Args:
            documents (list[str]): list of documents
            source (NamespaceKey): source namespace
        """

        def __get_metadata(doc) -> DocMetadata:
            return {
                **source._asdict(),
                "retrieval_date": self.retrieval_date.isoformat(),
            }

        index = create_index(
            INDEX_NAME,
            documents,
            index_impl=self.index_impl,
            get_doc_metadata=__get_metadata,
            context_args=self.context_args,
        )
        self.index = index

    def query(
        self,
        query_string: str,
        source: NamespaceKey,
        prompt: Optional[Prompt] = None,
        refine_prompt: Optional[RefinePrompt] = None,
    ) -> str:
        """
        Query the index

        Args:
            query_string (str): query string
            prompt (Prompt, optional): prompt. Defaults to None.
            refine_prompt (RefinePrompt, optional): refine prompt. Defaults to None.
        """
        if not self.index:
            raise ValueError("Index not initialized.")

        metadata_filters = get_metadata_filters(source)

        answer = query_index(
            self.index,
            query_string,
            prompt=prompt,
            refine_prompt=refine_prompt,
            metadata_filters=metadata_filters,
        )

        return answer
