"""
Biomedical Named Entity Recognition Keyword Table Index
"""
import logging
from typing import Any, Set
from llama_index import (
    QueryBundle,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table import SimpleKeywordTableIndex
from llama_index.indices.keyword_table.retrievers import BaseKeywordTableRetriever
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import NodeWithScore

from core.ner import NerTagger


def extract_keywords(
    text: str,
    tagger: NerTagger,
    max_keywords: int,
) -> set[str]:
    """
    Extract keywords from text using NER tagger.

    Args:
        text (str): text to extract keywords from
        tagger (NerTagger): NER tagger
        max_keywords (int): max number of keywords to extract
    """
    entities = tagger.extract_strings([text], link=False)
    keywords = entities[0]
    return set(keywords[0:max_keywords])


class KeywordTableNerRetriever(BaseKeywordTableRetriever):
    """
    Keyword Table Retriever with NER
    """

    def __init__(
        self,
        *args,
        tagger: NerTagger,
        **kwargs,
    ):
        """
        Initialize super and NER tagger.
        """
        super().__init__(*args, **kwargs)
        self.tagger = tagger

    def _get_keywords(self, query_str: str) -> list[str]:
        return list(
            extract_keywords(
                query_str,
                tagger=self.tagger,
                max_keywords=self.max_keywords_per_query,
            )
        )


class HybridNerRetriever(BaseRetriever):
    """
    Hybrid vector/keyword table retriever
    """

    def __init__(
        self,
        keyword_retriever: KeywordTableNerRetriever,
        vector_retriever: VectorIndexRetriever,
    ):
        """
        Initialize super and NER tagger.
        """
        super().__init__()
        self.keyword_retriever: KeywordTableNerRetriever = keyword_retriever
        self.vector_retriever: VectorIndexRetriever = vector_retriever
        self.mode = "OR"

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        keyword_nodes = self.keyword_retriever.retrieve(query_bundle)
        vector_nodes = self.vector_retriever.retrieve(query_bundle)

        logging.info(
            "Retrieved %s keyword nodes and %s vector nodes",
            len(keyword_nodes),
            len(vector_nodes),
        )

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self.mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes


class NerKeywordTableIndex(SimpleKeywordTableIndex):
    """
    Biomedical Named Entity Recognition Keyword Table Index
    """

    def __init__(
        self,
        *args,
        service_context: ServiceContext,
        storage_context: StorageContext,
        ner_options: dict[str, Any],
        **kwargs,
    ):
        """
        Initialize super and NER tagger.
        """
        super().__init__(
            *args,
            service_context=service_context,
            storage_context=storage_context,
            **kwargs,
        )
        self.tagger = NerTagger.get_instance(**ner_options, parallelize=False)
        self.vector_index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store, service_context
        )

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        raise NotImplementedError("Async not implemented for NER index")

    def _extract_keywords(self, text: str, max_keywords: int = 10000) -> Set[str]:
        """
        Extract keywords from text, using the simple method augmented by our biomedical NER.

        Args:
            text (str): text from which to extract keywords
            max_keywords (int, optional): maximum number of keywords to extract. Defaults to 10000.
        """
        return extract_keywords(text, tagger=self.tagger, max_keywords=max_keywords)

    def as_retriever(
        self,
        retriever_mode: str = "ner",
        **kwargs,
    ) -> HybridNerRetriever:
        if retriever_mode == "ner":
            vector_retriever = VectorIndexRetriever(
                index=self.vector_index, similarity_top_k=2
            )
            keyword_retriever = KeywordTableNerRetriever(
                self,
                tagger=self.tagger,
                vector_retriever=vector_retriever,
                **kwargs,
            )
            return HybridNerRetriever(
                keyword_retriever=keyword_retriever, vector_retriever=vector_retriever
            )
        else:
            raise ValueError(
                f"Unsupported retriever mode for custom index: {retriever_mode}"
            )

    def as_query_engine(self, **kwargs: Any):
        retriever = self.as_retriever(**kwargs)
        metadata_filters = kwargs.pop("metadata_filters")  # TODO

        response_synthesizer = get_response_synthesizer(
            service_context=self.service_context, **kwargs
        )
        nerkt_query_engine = RetrieverQueryEngine(
            retriever=retriever, response_synthesizer=response_synthesizer
        )

        return nerkt_query_engine

    def as_chat_engine(
        self, chat_mode: ChatMode = ChatMode.BEST, **kwargs: Any
    ) -> BaseChatEngine:
        logging.warning("Chat engine is just generic keyword table index, not NER")
        return super().as_chat_engine(chat_mode, **kwargs)

    def refresh_ref_docs(self, documents, **update_kwargs: Any) -> list[bool]:
        res = super().refresh_ref_docs(documents, **update_kwargs)
        self.vector_index.refresh_ref_docs(documents, **update_kwargs)
        return res
