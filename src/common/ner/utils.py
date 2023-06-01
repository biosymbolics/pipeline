"""
Named-entity recognition using spacy
"""
from typing import Literal
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex
import logging
from spacy_html_tokenizer import create_html_tokenizer


def __add_tokenization_re(
    nlp: Language,
    re_type: Literal["infixes", "prefixes", "suffixes"],
    new_res: list[str],
) -> list[str]:
    """
    Add regex to the tokenizer suffixes
    """
    if hasattr(nlp.Defaults, re_type):
        tokenizer_re_strs = list(getattr(nlp.Defaults, re_type))
        tokenizer_re_strs = [*tokenizer_re_strs, *new_res]
        return tokenizer_re_strs

    logging.warning(f"Could not find {re_type} in nlp.Defaults")
    return new_res


def __inner_html_tokenizer(nlp: Language) -> Tokenizer:
    """
    Add custom tokenization rules to the spacy tokenizer
    """
    prefix_re = __add_tokenization_re(nlp, "prefixes", ["•", "--", "/"])
    tokenizer = nlp.tokenizer
    tokenizer.prefix_search = compile_prefix_regex(prefix_re).search
    return tokenizer


UNWRAP_TAGS = ["em", "strong", "b", "i", "span", "a", "code", "kbd", "li"]


def get_sec_tokenizer(nlp: Language) -> Tokenizer:
    """
    Get the tokenizer for the sec pipeline
    (Handles HTML and some SEC-specific idiosyncracies)

    Args:
        nlp (Language): spacy language model
    """
    nlp.tokenizer = __inner_html_tokenizer(nlp)
    return create_html_tokenizer(unwrap_tags=UNWRAP_TAGS)(nlp)
