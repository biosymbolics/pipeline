"""
Utils for the NER pipeline
"""
from typing import Callable, Literal
import re
from functools import reduce
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.vocab import Vocab
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
import logging
import string

from common.utils.list import dedup
from common.utils.re import (
    get_or_re,
    ReCount,
    ALPHA_CHARS,
)
from common.utils.string import remove_unmatched_brackets

from .tokenizers.html_tokenizer import create_html_tokenizer


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


def remove_common_terms(
    vocab: Vocab, entity_list: list[str], exception_list: list[str] = []
):
    """
    Remove common terms from a list of entities, e.g. "vaccine candidates"

    Args:
        vocab (Vocab): spacy vocab
        entity_list (list[str]): list of entities
        exception_list (list[str]): list of exceptions to the common terms
    """

    def __is_common(item):
        # remove punctuation and make lowercase
        item_clean = item.lower().translate(str.maketrans("", "", string.punctuation))
        words = item_clean.split()

        # check if all words are in the vocab
        common = set(words).issubset(vocab.strings)

        # check if any words are in the exception list
        excepted = len(set(words).intersection(set(exception_list))) == 0

        if common and not excepted:
            logging.info(f"Removing common term: {item}")
        elif excepted:
            logging.info(f"Keeping exception term: {item}")
        return common

    def __is_uncommon(item):
        return not __is_common(item)

    return list(filter(__is_uncommon, entity_list))


CHARACTER_SUPPRESSIONS = [r"\n", "®"]


def clean_entity(entity: str) -> str:
    """
    Clean entity name
    - remove certain characters
    - removes double+ spaces
    - removed unmatched () [] {} <>

    Args:
        entity (str): entity name
    """
    removal_pattern = get_or_re(CHARACTER_SUPPRESSIONS)

    def remove_characters(entity: str) -> str:
        return re.sub(removal_pattern, " ", entity)

    def remove_extra_spaces(entity: str) -> str:
        return re.sub(r"\s+", " ", entity.strip())

    # List of cleaning functions to apply to entity
    cleaning_steps = [remove_characters, remove_extra_spaces, remove_unmatched_brackets]

    cleaned = reduce(lambda x, func: func(x), cleaning_steps, entity)

    if cleaned != entity:
        logging.debug(f"Cleaned entity: {entity} -> {cleaned}")

    return cleaned


INCLUSION_SUPPRESSIONS = ["phase", "trial"]


CleaningFunction = Callable[[list[str]], list[str]]


def sanitize_entity_names(entity_map: dict[str, list[str]], nlp: Language) -> list[str]:
    """
    Clean entity name list

    Args:
        entity_map (dict[str, list[str]]): entity map
    """

    def __filter_entities(entity_names: list[str]) -> list[str]:
        """
        Filter out entities that are not relevant
        """

        def __filter(entity: str) -> bool:
            """
            Filter out entities that are not relevant

            Args:
                entity (str): entity name
            """
            is_suppressed = any(
                [sup in entity.lower() for sup in INCLUSION_SUPPRESSIONS]
            )
            return not is_suppressed

        filtered = [entity for entity in entity_names if __filter(entity)]
        without_common = remove_common_terms(nlp.vocab, filtered)
        return dedup(without_common)

    def __clean_entities(entity_names: list[str]) -> list[str]:
        """
        Clean entity name list
        """
        return [clean_entity(entity) for entity in entity_names]

    cleaning_steps: list[CleaningFunction] = [
        __filter_entities,
        dedup,
        __clean_entities,
    ]
    entity_names = list(entity_map.keys())

    sanitized = reduce(lambda x, func: func(x), cleaning_steps, entity_names)

    return sanitized
