"""
SourceDocIndex
"""
from typing import Optional
from llama_index import GPTKeywordTableIndex

from clients.llama_index import create_index, get_composed_index, query_index
from constants.core import DEFAULT_MODEL_NAME
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
        index_impl: LlmIndex = GPTKeywordTableIndex,  # type: ignore
        model_name: LlmModelType = DEFAULT_MODEL_NAME,
    ):
        """
        initialize

        Args:
            source: NamespaceKey
            index_id: str
            index_impl: LlmIndex
            model_name: LlmModelType
        """
        self.index_impl = index_impl
        self.source = source
        self.index_id = index_id
        self.index = None
        self.model_name = model_name

    def __create__(self, documents):
        """
        Create index

        Args:
            documents: list[str]
        """
        index = create_index(
            self.source,
            self.index_id,
            documents,
            index_impl=self.index_impl,
            model_name=self.model_name,  # type: ignore
        )
        return index

    def load(self, documents: list[str]):
        """
        Load docs into index
        TODO: ability to add docs?

        Args:
            documents: list[str]
        """
        self.index = self.__create__(documents)

    def query(
        self,
        query_string: str,
        prompt: Optional[Prompt] = None,
        refine_prompt: Optional[RefinePrompt] = None,
    ) -> str:
        """
        Query the index

        Args:
            query_string: str
            prompt: Optional[Prompt]
            refine_prompt: Optional[RefinePrompt]
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
      - ["BIBB", "SEC", "10-K"] will query over all years of BIBB (Biogen) 10-K docs (year is index_id)
      - ["BIBB", "SEC", "8-K"] will query over all years of BIBB 8-K docs
      - ["BIBB", "SEC"] will query over all years all SEC docs

    TODO: ability to pull, for example, "all 10-K docs mentioning X" (regardless of company)

    NOTE: can accept a different model_name than the constituent indices (used for composite index formation)
    """

    def __init__(
        self, source: NamespaceKey, model_name: LlmModelType = DEFAULT_MODEL_NAME
    ):
        """
        initialize

        Args:
            source: NamespaceKey
            model_name: LlmModelType
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
            query_string: str
            prompt: Optional[Prompt]
            refine_prompt: Optional[RefinePrompt]
        """
        if not self.index:
            raise ValueError("No index found.")

        answer = query_index(
            self.index, query_string, prompt=prompt, refine_prompt=refine_prompt
        )
        return answer
