"""
Utils for the NER pipeline
"""
from typing import Literal
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import logging
import string

from .tokenizers.html_tokenizer import create_html_tokenizer

from common.utils.re import (
    get_or_re,
    ReCount,
    ALPHA_CHARS,
)


def __add_tokenization_re(
    nlp: Language,
    re_type: Literal["infixes", "prefixes", "suffixes"],
    new_res: list[str],
) -> list[str]:
    """
    Add regex to the tokenizer suffixes
    """
    if hasattr(nlp.Defaults, re_type):
        tokenizer_re_strs = getattr(nlp.Defaults, re_type) + new_res
        return tokenizer_re_strs

    logging.warning(f"Could not find {re_type} in nlp.Defaults")
    return new_res


def __inner_html_tokenizer(nlp: Language) -> Tokenizer:
    """
    Add custom tokenization rules to the spacy tokenizer
    """
    prefix_re = __add_tokenization_re(nlp, "prefixes", ["•", "——"])
    suffix_re = __add_tokenization_re(nlp, "suffixes", [":"])
    # infix_re = __add_tokenization_re(nlp, "infixes", ["\\+"])
    tokenizer = nlp.tokenizer
    tokenizer.prefix_search = compile_prefix_regex(prefix_re).search
    tokenizer.suffix_search = compile_suffix_regex(suffix_re).search
    # tokenizer.infix_finditer = compile_infix_regex(infix_re).finditer
    return tokenizer


def get_sec_tokenizer(nlp: Language) -> Tokenizer:
    """
    Get the tokenizer for the sec pipeline
    (Handles HTML and some SEC-specific idiosyncracies)

    Args:
        nlp (Language): spacy language model
    """
    nlp.tokenizer = __inner_html_tokenizer(nlp)
    return create_html_tokenizer()(nlp)


"""
Re utils
"""
# end-of-entity regex
EOE_RE = "\\b" + ".*"

# start-of-entity regex
SOE_RE = ".*"


def get_entity_re(
    core_entity_re: str,
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    is_case_insensitive=False,
) -> str:
    """
    Returns a regex for an entity with a start-of-entity (soe_re) and end-of-entity (eoe_re) regexes

    Args:
        core_entity_re (str): regex for the core entity
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        is_case_insensitive (bool): whether to make the regex case insensitive (default: False)
    """
    core_re = soe_re + core_entity_re + eoe_re

    if is_case_insensitive:
        return "(?i)" + core_re

    return core_re


def get_infix_entity_re(
    core_infix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_infix_res (list[str]): list of regexes for infixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        count (ReCount): number of alpha chars in prefix and suffix
    """
    return (
        soe_re
        + ALPHA_CHARS(count)
        + get_or_re(core_infix_res)
        + ALPHA_CHARS(count)
        + eoe_re
    )


def get_suffix_entitiy_re(
    core_suffix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    prefix_count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_suffix_res (list[str]): list of regexes for suffixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        prefix_count (ReCount): number of alpha chars in prefix
    """
    return soe_re + ALPHA_CHARS(prefix_count) + get_or_re(core_suffix_res) + eoe_re


"""
Other utils
"""


def remove_common_terms(vocab: Vocab, entity_list: list[str]):
    """
    Remove common terms from a list of entities

    Args:
        vocab (Vocab): spacy vocab
        entity_list (list[str]): list of entities
    """

    def is_all_common(item):
        item_clean = item.lower().translate(str.maketrans("", "", string.punctuation))
        words = item_clean.split()
        all_common = set(words).issubset(vocab.strings)
        if all_common:
            logging.info(f"Removing common term: {item}")
        return all_common

    return list(filter(is_all_common, entity_list))
