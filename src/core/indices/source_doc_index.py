"""
SourceDocIndex
"""
from datetime import datetime
from typing import Optional
from llama_index import GPTKeywordTableIndex

from clients.llama_index import create_index, get_composed_index, query_index
from clients.llama_index.context import ContextArgs, DEFAULT_CONTEXT_ARGS
from clients.llama_index.types import DocMetadata
from types.indices import LlmIndex, LlmModelType, NamespaceKey, Prompt, RefinePrompt


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs
    """

    def __init__(
        self,
        source: NamespaceKey,
        index_id: str,
        context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
        documents: Optional[list[str]] = None,
        index_impl: LlmIndex = GPTKeywordTableIndex,  # type: ignore
        retrieval_date: datetime = datetime.now(),
    ):
        """
        initialize

        Args:
            source (NamespaceKey): source of the index
            index_id (str): id of the index
            context_args (ContextArgs, optional): context args. Defaults to DEFAULT_CONTEXT_ARGS.
            documents (list[str], optional): list of documents. Defaults to None. Can be loaded later with from_documents().
            index_impl (LlmIndex, optional): index implementation. Defaults to GPTKeywordTableIndex.
            retrieval_date (datetime, optional): retrieval date of source docs. Defaults to datetime.now().
        """
        self.context_args = context_args
        self.index = None  # TODO: load
        self.index_impl = index_impl
        self.index_id = index_id
        self.source = source
        self.retrieval_date = retrieval_date

        # if docs provided, load
        if documents:
            self.from_documents(documents)

    def from_documents(self, documents: list[str]):
        """
        Load docs into index with metadata

        Args:
            documents: list[str]
        """

        def __get_metadata(doc) -> DocMetadata:
            return {
                "retrieval_date": self.retrieval_date.isoformat(),
                **self.source._asdict(),
            }

        index = create_index(
            self.source,
            self.index_id,
            documents,
            index_impl=self.index_impl,
            get_doc_metadata=__get_metadata,
            context_args=self.context_args,
        )
        self.index = index

    def query(
        self,
        query_string: str,
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

        answer = query_index(
            self.index,
            query_string,
            prompt=prompt,
            refine_prompt=refine_prompt,
        )

        return answer


class CompositeSourceDocIndex:
    """
    CompositeSourceDocIndex

    Can accept "source" at any level.
    For example:
      - (company="BIBB", doc_source="SEC", doc_type="10-K") will query over all years of BIBB (Biogen) 10-K docs (year is index_id)
      - (company="BIBB", doc_source="SEC", doc_type="8-K") will query over all years of BIBB 8-K docs
      - (company="BIBB", doc_type="SEC") will query over all years all SEC docs

    TODO: ability to pull, for example, "all 10-K docs mentioning X" (regardless of company)

    NOTE: can accept a different model_name than the constituent indices (used for composite index formation)
    """

    def __init__(
        self,
        source: NamespaceKey,
        context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
    ):
        """
        initialize

        Args:
            source (NamespaceKey): source of the index
            context_args (ContextArgs, optional): context args. Defaults to DEFAULT_CONTEXT_ARGS.
        """
        self.source = source
        self.index = get_composed_index(source, context_args)

    def query(
        self,
        query_string: str,
        prompt: Optional[Prompt] = None,
        refine_prompt: Optional[RefinePrompt] = None,
    ) -> str:
        """
        Query the composite index

        Args:
            query_string (str): query string
            prompt (Prompt, optional): prompt. Defaults to None.
            refine_prompt (RefinePrompt, optional): refine prompt. Defaults to None.
        """
        if not self.index:
            raise ValueError("No index found.")

        answer = query_index(
            self.index, query_string, prompt=prompt, refine_prompt=refine_prompt
        )
        return answer
