"""
Biomedical Named Entity Recognition Keyword Table Index
"""
import logging
from typing import Optional, Set
from llama_index.indices.keyword_table import SimpleKeywordTableIndex
from pydash import flatten

from common.ner.ner import NerTagger


class NerKeywordTableIndex(SimpleKeywordTableIndex):
    """
    Biomedical Named Entity Recognition Keyword Table Index
    """

    def __init__(
        self,
        *args,
        ner_options={
            "use_llm": True,
            "content_type": "html",
            "llm_config": "configs/sec/config.cfg",
        },
        **kwargs,
    ):
        """
        Initialize super and NER tagger.
        """
        super().__init__(*args, **kwargs)
        self.tagger = NerTagger.get_instance(**ner_options)

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
            [(ent[0], ent[2].name) if ent[2] else (ent[0]) for ent in entities]
        )
        return set(keywords[0:max_keywords])

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        Extract keywords from text, using the simple method augmented by our biomedical NER.

        Args:
            text (str): text from which to extract keywords
        """

        simple_keywords = super()._extract_keywords(text)
        ner_keywords = self._ner_extract_keywords(
            text, max_keywords=self.max_keywords_per_chunk
        )
        return set([*ner_keywords, *simple_keywords])
