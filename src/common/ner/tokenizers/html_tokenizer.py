"""
Tokenization based on HTML tags

Loosely based on  https://github.com/pmbaumgartner/spacy-html-tokenizer
"""
from typing import Callable, List, Union

from spacy.util import registry
from spacy.tokens import Doc
from selectolax.parser import HTMLParser, Node
from spacy.tokenizer import Tokenizer
from pydash import compact

DEFAULT_REMOVE_TAGS: list[str] = ["script", "style", "hr", "br"]


def join_strings(texts: list[str], separator: str = " ") -> str:
    return separator.join(compact(texts))


def table_to_text(node: Node) -> str:
    rows = node.css("tr")
    return join_strings([row_to_text(row) for row in rows], ".\n ")


def row_to_text(node: Node) -> str:
    cells = node.css("td")
    return join_strings(
        [text_node_to_text(cell, ", ") for cell in cells], ", "
    )  # TODO: sep only ", " for short cells?


def text_node_to_text(node: Node, separator: str = " ") -> str:
    node_text = node.text(deep=True, strip=False, separator=separator)
    node_text = node_text.strip()
    return node_text


def element_to_text(node: Node) -> str:
    texts: list[str] = []
    if node.tag == "table":
        texts = [*texts, table_to_text(node)]
    elif node.tag == "tr":
        texts = [*texts, row_to_text(node)]
    elif node.tag == "div":
        children = [
            child
            for child in node.css("*")
            if not child.__eq__(node) and child.parent == node
        ]
        if len(children) == 0:
            texts = [*texts, text_node_to_text(node)]
        else:
            texts = [*texts, *[element_to_text(child) for child in children]]
    else:
        texts = [*texts, text_node_to_text(node)]

    return join_strings(texts)


class HTMLTokenizer(Tokenizer):
    def __init__(self, remove_tags: List[str] = DEFAULT_REMOVE_TAGS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.remove_tags = remove_tags

    def __call_super__(self, string: str) -> Doc:
        return super().__call__(string)

    def __call__(self, string) -> Doc:
        html_texts = self.parse_html(string)
        html_docs = [self.__call_super__(text) for text in html_texts]
        doc = Doc.from_docs(html_docs)
        return doc

    def parse_html(self, html_string: str) -> List[str]:
        parsed_html = HTMLParser(html_string)
        for removed_tag in self.remove_tags:
            for element in parsed_html.css(removed_tag):
                element.decompose()

        top_level_nodes = parsed_html.root.css("body > *") if parsed_html.root else []
        html_texts = [element_to_text(node) for node in top_level_nodes]
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
