"""
SourceDocIndex
"""
from datetime import datetime
from typing import Optional
from llama_index import GPTKeywordTableIndex

from clients.llama_index import create_index, get_composed_index, query_index
from constants.core import DEFAULT_MODEL_NAME
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
        documents: Optional[list[str]] = None,
        retrieval_date: datetime = datetime.now(),
        index_impl: LlmIndex = GPTKeywordTableIndex,  # type: ignore
        model_name: LlmModelType = DEFAULT_MODEL_NAME,
    ):
        """
        initialize

        Args:
            source (NamespaceKey): source of the index
            index_id (str): id of the index
            retrieval_date (datetime, optional): retrieval date of source doc
            index_impl (LlmIndex, optional): index implementation. Defaults to GPTKeywordTableIndex.
            model_name (LlmModelType): model name. Defaults to DEFAULT_MODEL_NAME
        """
        self.index = None  # TODO: load
        self.index_impl = index_impl
        self.index_id = index_id
        self.source = source
        self.model_name: LlmModelType = model_name
        self.retrieval_date = retrieval_date

        # if docs provided, load
        if documents:
            self.from_documents(documents)

    def from_documents(self, documents: list[str]):
        """
        Load docs into index

        Args:
            documents: list[str]
        """

        def __get_metadata(doc) -> DocMetadata:
            # TODO: informative names for keys
            source_info = {
                f"source{index+1}": value for index, value in enumerate(self.source)
            }
            return {"retrieval_date": self.retrieval_date.isoformat(), **source_info}

        index = create_index(
            self.source,
            self.index_id,
            documents,
            index_impl=self.index_impl,
            model_name=self.model_name,
            get_doc_metadata=__get_metadata,
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
        self, source: NamespaceKey, model_name: LlmModelType = DEFAULT_MODEL_NAME
    ):
        """
        initialize

        Args:
            source (NamespaceKey): source of the index
            model_name (LlmModelType): model name. Defaults to DEFAULT_MODEL_NAME
        """
        self.source = source
        self.index = get_composed_index(source, model_name)

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
