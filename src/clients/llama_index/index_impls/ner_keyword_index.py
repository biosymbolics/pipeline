from typing import Optional, Set

from llama_index.indices.keyword_table.base import BaseKeywordTableIndex
from pydash import flatten

from common.ner.ner import NerTagger


class BNerKeywordTableIndex(BaseKeywordTableIndex):
    """
    Biomedical Named Entity Recognition Keyword Table Index
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize super and NER tagger.
        """
        super().__init__(*args, **kwargs)
        self.tagger = NerTagger.get_instance(use_llm=True, content_type="text")

    def _ner_extract_keywords(
        self, text: str, max_keywords: Optional[int] = 10000
    ) -> Set[str]:
        """
        Extract keywords from text using biomedical NER.
        Includes original term plus canonical term if available.

        Args:
            text (str): text to extract keywords from
            max_keywords (Optional[int], optional): maximum number of keywords to extract. Defaults to 10000.
        """
        entities = self.tagger.extract([text], link=True)[0]
        keywords = flatten(
            [(ent[0], ent[2].name) if ent[2] else (ent[0]) for ent in entities]
        )
        return set(keywords[0:max_keywords])

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        return self._ner_extract_keywords(
            text, max_keywords=self.max_keywords_per_chunk
        )
