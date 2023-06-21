"""
Utils for the NER pipeline
"""
from typing import Callable
import re
from functools import reduce
from spacy.language import Language
from spacy.vocab import Vocab
import logging
import string

from common.utils.list import dedup
from common.utils.re import get_or_re, remove_extra_spaces, LEGAL_SYMBOLS

CHAR_SUPPRESSIONS = [r"\n", *LEGAL_SYMBOLS]
INCLUSION_SUPPRESSIONS = ["phase", "trial"]

CleaningFunction = Callable[[list[str]], list[str]]


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
        is_common = set(words).issubset(vocab.strings)

        # check if any words are in the exception list
        is_excepted = (
            len(set(exception_list).intersection(set(words))) > 0 or len(words) > 1
        )  # HACK!

        is_common_not_excepted = is_common and not is_excepted

        if is_common_not_excepted:
            logging.info(f"Removing common term: {item}")
        elif is_excepted:
            logging.info(f"Keeping exception term: {item}")
        return is_common_not_excepted

    def __is_uncommon(item):
        return not __is_common(item)

    return list(filter(__is_uncommon, entity_list))


def clean_entity(entity: str, char_suppressions: list[str] = CHAR_SUPPRESSIONS) -> str:
    """
    Clean entity name
    - remove certain characters
    - removes double+ spaces
    - removed unmatched () [] {} <>

    Args:
        entity (str): entity name
    """
    removal_pattern = get_or_re(char_suppressions)

    def remove_characters(entity: str) -> str:
        return re.sub(removal_pattern, " ", entity)

    # List of cleaning functions to apply to entity
    cleaning_steps = [remove_characters, remove_extra_spaces]

    cleaned = reduce(lambda x, func: func(x), cleaning_steps, entity)

    if cleaned != entity:
        logging.info(f"Cleaned entity: {entity} -> {cleaned}")

    return cleaned


def clean_entities(entities: list[str], nlp: Language) -> list[str]:
    """
    Clean entity name list
    - filters out (some) excessively general entities
    - dedups
    - cleans entity names

    Args:
        entities (list[str]): entities map
        nlp (Language): spacy language model
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
        without_common = remove_common_terms(nlp.vocab, filtered)  # disabling for now
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

    sanitized = reduce(lambda x, func: func(x), cleaning_steps, entities)

    return sanitized
