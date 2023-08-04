"""
Biomedical Named Entity Recognition Keyword Table Index
"""
import logging
from typing import Any, Optional, Set
from llama_index.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.indices.keyword_table import SimpleKeywordTableIndex
from llama_index.indices.keyword_table.retrievers import BaseKeywordTableRetriever
from llama_index import get_response_synthesizer
from llama_index.query_engine.retriever_query_engine import RetrieverQueryEngine
from pydash import flatten

from common.ner.ner import NerTagger


def extract_keywords(text: str, tagger: NerTagger, max_keywords: int) -> set[str]:
    """
    Extract keywords from text using NER tagger.
    """
    entities_by_doc = tagger.extract([text], link=True)

    logging.info(f"Extracted {len(entities_by_doc)} docs")

    if len(entities_by_doc) == 0:
        raise ValueError("No entities extracted")

    entities = entities_by_doc[0]
    keywords = flatten(
        [
            (ent.term, ent.linked_entity.name)
            if ent.linked_entity
            else (ent.normalized_term)
            for ent in entities
        ]
    )
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
                query_str, tagger=self.tagger, max_keywords=self.max_keywords_per_query
            )
        )


class NerKeywordTableIndex(SimpleKeywordTableIndex):
    """
    Biomedical Named Entity Recognition Keyword Table Index
    """

    def __init__(
        self,
        *args,
        ner_options={"use_llm": False},
        **kwargs,
    ):
        """
        Initialize super and NER tagger.
        """
        super().__init__(*args, **kwargs)
        self.tagger = NerTagger.get_instance(**ner_options, parallelize=False)
        self.response_synthesizer = get_response_synthesizer()

    def _ner_extract_keywords(
        self, text: str, max_keywords: Optional[int] = 10000
    ) -> Set[str]:
        """
        Extract keywords from text using biomedical NER.
        Includes original term plus canonical term if available.

        Args:
            text (str): text from which to extract keywords
            max_keywords (Optional[int], optional): maximum number of keywords to extract. Defaults to 10000.
        """
        entities_by_doc = self.tagger.extract([text], link=True)

        logging.info(f"Extracted {len(entities_by_doc)} docs")

        if len(entities_by_doc) == 0:
            raise ValueError("No entities extracted")

        entities = entities_by_doc[0]
        keywords = flatten(
            [
                (ent.term, ent.linked_entity.name)
                if ent.linked_entity
                else (ent.normalized_term)
                for ent in entities
            ]
        )
        return set(keywords[0:max_keywords])

    async def _async_extract_keywords(self, text: str) -> Set[str]:
        raise NotImplementedError("Async not implemented for NER index")

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text, using the simple method augmented by our biomedical NER.

        Args:
            text (str): text from which to extract keywords
        """

        keywords = self._ner_extract_keywords(
            text, max_keywords=self.max_keywords_per_chunk
        )
        return keywords

    def as_retriever(
        self,
        retriever_mode: str = "ner",
        **kwargs,
    ) -> KeywordTableNerRetriever:
        if retriever_mode == "ner":
            return KeywordTableNerRetriever(self, tagger=self.tagger, **kwargs)
        else:
            raise ValueError(
                f"Unsupported retriever mode for custom index: {retriever_mode}"
            )

    def as_query_engine(self, **kwargs: Any):
        retriever = self.as_retriever(**kwargs)

        ktner_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=self.response_synthesizer,
            # **kwargs,
        )

        return ktner_query_engine

    def as_chat_engine(
        self, chat_mode: ChatMode = ChatMode.BEST, **kwargs: Any
    ) -> BaseChatEngine:
        logging.warning("Chat engine is just generic keyword table index, not NER")
        return super().as_chat_engine(chat_mode, **kwargs)
