"""
Utils for the NER pipeline
"""
from typing import Callable, TypeVar, Union, cast
import re
from functools import partial, reduce
from spacy.language import Language
from spacy.tokens import Span
import logging

from common.utils.list import dedup
from common.utils.re import remove_extra_spaces, LEGAL_SYMBOLS

CHAR_SUPPRESSIONS = {
    r"\n": " ",
    "/": " ",
    **{symbol: "" for symbol in LEGAL_SYMBOLS},
}
INCLUSION_SUPPRESSIONS = ["phase", "trial"]

T = TypeVar("T", bound=Union[Span, str])
CleanFunction = Callable[[list[T]], list[T]]

DEFAULT_EXCEPTION_LIST: list[str] = [
    "hiv",
    "asthma",
    "obesity",
    "covid",
    "diabetes",
    "kidney",
    "liver",
    "heart",
    "lung",
    "cancer",
    "arthritis",
    "stroke",
    "dementia",
    "trauma",
    "insulin",
    "depression",
    "anxiety",
    "g",  # g protein coupled receptor
    "pain",
    "cardiovascular",
    "respiratory",
    "aging",
]

DEFAULT_ADDITIONAL_COMMON_WORDS = [
    "(i)",  # so common in patents, e.g. "general formula (I)"
]


def get_common_words(additional_words):
    """
    Get common words from a file + additional common words
    """
    with open("10000words.txt", "r") as file:
        vocab_words = file.read().splitlines()
    return [*vocab_words, *additional_words]


def remove_common(
    entity_list: list[T],
    nlp: Language,
    exception_list: list[str] = DEFAULT_EXCEPTION_LIST,
    additional_common_words: list[str] = DEFAULT_ADDITIONAL_COMMON_WORDS,
) -> list[T]:
    """
    Remove common terms from a list of entities, e.g. "vaccine candidates"

    Args:
        entity_list (list[T]): list of entities
        nlp (Language): spacy language model
        exception_list (list[str]): list of exceptions to the common terms
    """

    common_words = get_common_words(additional_common_words)

    def __is_common(item: T):
        name = item.text if isinstance(item, Span) else item

        # remove punctuation and make lowercase
        words = [token.lemma_ for token in nlp(name)]

        # check if all words are in the vocab
        is_common = set(words).issubset(common_words)

        # check if any words are in the exception list
        is_excepted = bool(set(exception_list) & set(words))

        is_common_not_excepted = is_common and not is_excepted

        if is_common_not_excepted:
            logging.info(f"Removing common term: {item}")
        elif is_excepted:
            logging.info(f"Keeping exception term: {item}")
        return is_common_not_excepted

    def __is_uncommon(item):
        return not __is_common(item)

    return list(filter(__is_uncommon, entity_list))


def normalize_entity(
    entity: T, char_suppressions: dict[str, str] = CHAR_SUPPRESSIONS
) -> T:
    """
    Normalize entity name
    - remove certain characters
    - removes double+ spaces

    Args:
        entity (T): entity
    """
    name = entity.text if isinstance(entity, Span) else entity

    def remove_chars(entity_name: str) -> str:
        for pattern, replacement in char_suppressions.items():
            entity_name = re.sub(pattern, replacement, entity_name)
        return entity_name

    cleaning_steps = [remove_chars, remove_extra_spaces]

    normalized = reduce(lambda x, func: func(x), cleaning_steps, name)

    if normalized != name:
        logging.info(f"Normalized entity: {name} -> {normalized}")

    if isinstance(entity, Span):
        return cast(T, Span(entity.doc, entity.start, entity.end, label=entity.label))

    return normalized


def normalize_entities(entities: list[T]) -> list[T]:
    """
    normalize entity names

    Args:
        entities (list[T]): entities
    """
    return [normalize_entity(entity) for entity in entities]


def sanitize_entities(entities: list[T], nlp: Language) -> list[T]:
    """
    Sanitize entity list
    - filters out (some) excessively general entities
    - dedups
    - normalizes entity names

    Args:
        entities (list[T]): entities
        nlp (Language): spacy language model
    """

    def suppress(entities: list[T]) -> list[T]:
        """
        Filter out irrelevant entities
        """

        def should_keep(entity: T) -> bool:
            name = entity.text if isinstance(entity, Span) else entity
            is_suppressed = any([sup in name.lower() for sup in INCLUSION_SUPPRESSIONS])
            return not is_suppressed

        return [entity for entity in entities if should_keep(entity)]

    cleaning_steps: list[CleanFunction] = [
        suppress,
        partial(remove_common, nlp=nlp),
        normalize_entities,
        dedup,
    ]

    sanitized = reduce(lambda x, func: func(x), cleaning_steps, entities)

    return sanitized
