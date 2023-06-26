"""
Tokenization based on HTML tags

Loosely based on  https://github.com/pmbaumgartner/spacy-html-tokenizer
"""
from typing import List

from spacy.util import registry
from spacy.tokens import Doc
from selectolax.parser import HTMLParser, Node
from spacy.tokenizer import Tokenizer
import logging

DEFAULT_REMOVE_TAGS: list[str] = ["script", "style", "hr", "br"]


class HTMLTokenizer(Tokenizer):
    """
    Tokenizer for HTML
    """

    def __init__(self, remove_tags: List[str] = DEFAULT_REMOVE_TAGS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tags = remove_tags

    def __call_super__(self, string: str) -> Doc:
        # needed for list comprehension?
        return super().__call__(string)

    def __call__(self, string) -> Doc:
        html_texts = self.parse_html(string)
        html_docs = [self.__call_super__(text) for text in html_texts]
        doc = Doc.from_docs(html_docs)
        return doc

    def __join(self, texts: list[str], separator: str = " ") -> str:
        """
        Join a list of strings with a separator
        """
        return separator.join(text for text in texts if text)

    def __table_to_text(self, node: Node) -> str:
        """
        Convert a table to text
        """
        rows = node.css("tr")
        return self.__join([self.__row_to_text(row) for row in rows], ".\n ")

    def __row_to_text(self, node: Node) -> str:
        """
        Convert a row to text
        """
        cells = node.css("td")
        return self.__join(
            [self.__text_node_to_text(cell, ", ") for cell in cells], ", "
        )

    def __text_node_to_text(self, node: Node, separator: str = " ") -> str:
        """
        Convert a text node to text
        """
        node_text = node.text(deep=True, strip=False, separator=separator)
        node_text = node_text.strip()
        return node_text

    def __element_to_text(self, node: Node) -> str:
        """
        Convert an element to text
        """
        if node.tag == "table":
            text = self.__table_to_text(node)
        elif node.tag == "tr":
            text = self.__row_to_text(node)
        elif node.tag == "div":
            children = [
                child
                for child in node.css("*")
                if not child.__eq__(node) and child.parent == node
            ]
            if len(children) == 0:
                text = self.__text_node_to_text(node)
            else:
                text = self.__join(
                    [self.__element_to_text(child) for child in children]
                )
        else:
            text = self.__text_node_to_text(node)

        return text

    def parse_html(self, html_string: str) -> List[str]:
        parsed_html = HTMLParser(html_string)
        for removed_tag in self.remove_tags:
            for element in parsed_html.css(removed_tag):
                element.decompose()

        top_level_nodes = parsed_html.root.css("body > *") if parsed_html.root else []
        html_texts = [self.__element_to_text(node) for node in top_level_nodes]
        return html_texts


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
