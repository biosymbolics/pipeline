"""
SourceDocIndex
"""
from llama_index import GPTKeywordTableIndex

from clients.llama_index import create_index, query_index
from types.indices import LlmIndex, NamespaceKey


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs
    """

    def __init__(self, source: NamespaceKey, index_id: str, index_impl: LlmIndex = GPTKeywordTableIndex):  # type: ignore
        self.index_impl = index_impl
        self.source = source
        self.index_id = index_id
        self.index = None

    def __create__(self, documents):
        """
        Create index
        """
        index = create_index(
            self.source,
            self.index_id,
            documents,
            index_impl=self.index_impl,
        )
        return index

    def load(self, documents: list[str]):
        """
        Load docs into index
        TODO: add docs
        """
        self.index = self.__create__(documents)

    def query(self, query_string: str):
        if not self.index:
            raise ValueError("Index not initialized.")

        answer = query_index(self.index, query_string)
        return answer
