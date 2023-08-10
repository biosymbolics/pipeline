"""
Tokenization based on HTML tags
"""
from typing import List

from spacy.util import registry
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
import logging

from utils.extraction.html import ContentExtractor, DEFAULT_REMOVE_TAGS


class HTMLTokenizer(Tokenizer):
    """
    Tokenizer for HTML
    """

    def __init__(self, remove_tags: List[str] = DEFAULT_REMOVE_TAGS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extractor = ContentExtractor(remove_tags=remove_tags)
        self.remove_tags = remove_tags

    def __call_super__(self, string: str) -> Doc:
        # needed for list comprehension?
        return super().__call__(string)

    def __call__(self, string) -> Doc:
        html_texts = self.extractor.parse_html(string)
        html_docs = [self.__call_super__(text) for text in html_texts]
        doc = Doc.from_docs(html_docs)
        return doc


@registry.tokenizers("html_tokenizer")
def create_html_tokenizer(
    remove_tags: List[str] = DEFAULT_REMOVE_TAGS,
):
    """
    Factory function for HTMLTokenizer
    """

    def create_tokenizer(nlp) -> Tokenizer:
        tokenizer = HTMLTokenizer(
            vocab=nlp.vocab,
            rules=nlp.tokenizer.rules,
            prefix_search=nlp.tokenizer.prefix_search,
            suffix_search=nlp.tokenizer.suffix_search,
            infix_finditer=nlp.tokenizer.infix_finditer,
            token_match=nlp.tokenizer.token_match,
            url_match=nlp.tokenizer.url_match,
            remove_tags=remove_tags,
        )
        return tokenizer

    return create_tokenizer
