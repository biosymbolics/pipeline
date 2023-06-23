"""
Utils for the NER pipeline
"""
from typing import Callable
import re
from functools import reduce
from spacy.language import Language
import logging

from common.utils.list import dedup
from common.utils.re import remove_extra_spaces, LEGAL_SYMBOLS

CHAR_SUPPRESSIONS = {
    r"\n": " ",
    "/": " ",
    **{symbol: "" for symbol in LEGAL_SYMBOLS},
}
INCLUSION_SUPPRESSIONS = ["phase", "trial"]

CleaningFunction = Callable[[list[str]], list[str]]

with open("10000words.txt", "r") as file:
    vocab_words = file.read().splitlines()


def remove_common_terms(
    entity_list: list[str], nlp: Language, exception_list: list[str] = []
):
    """
    Remove common terms from a list of entities, e.g. "vaccine candidates"

    Args:
        entity_list (list[str]): list of entities
        nlp (Language): spacy language model
        exception_list (list[str]): list of exceptions to the common terms
    """

    def __is_common(item):
        # remove punctuation and make lowercase
        words = [token.lemma_ for token in nlp(item)]

        # check if all words are in the vocab
        is_common = set(words).issubset(vocab_words)

        # check if any words are in the exception list
        is_excepted = len(set(exception_list).intersection(set(words))) > 0

        is_common_not_excepted = is_common and not is_excepted

        if is_common_not_excepted:
            logging.info(f"Removing common term: {item}")
        elif is_excepted:
            logging.info(f"Keeping exception term: {item}")
        return is_common_not_excepted

    def __is_uncommon(item):
        return not __is_common(item)

    return list(filter(__is_uncommon, entity_list))


def clean_entity(
    name: str, char_suppressions: dict[str, str] = CHAR_SUPPRESSIONS
) -> str:
    """
    Clean entity name
    - remove certain characters
    - removes double+ spaces

    Args:
        entity (str): entity name
    """

    def remove_characters(entity: str) -> str:
        for pattern, replacement in char_suppressions.items():
            entity = re.sub(pattern, replacement, entity)
        return entity

    # List of cleaning functions to apply to entity
    cleaning_steps = [remove_characters, remove_extra_spaces]

    cleaned = reduce(lambda x, func: func(x), cleaning_steps, name)

    if cleaned != name:
        logging.info(f"Cleaned entity: {name} -> {cleaned}")

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
        without_common = remove_common_terms(filtered, nlp)
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
