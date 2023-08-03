"""
Utils for the NER pipeline
"""
from typing import Callable, TypeVar, Union, cast
import re
from functools import partial, reduce
import logging
import html

from clients.spacy import Spacy
from common.ner.utils import lemmatize_tail
from common.utils.list import dedup
from common.utils.re import remove_extra_spaces, LEGAL_SYMBOLS
from typings.core import is_string_list

from .types import DocEntity, is_entity_doc_list
from .utils import normalize_by_pos, rearrange_terms

T = TypeVar("T", bound=Union[DocEntity, str])
CleanFunction = Callable[[list[T]], list[T]]

CHAR_SUPPRESSIONS = {
    r"\n": " ",
    "/": " ",
    r"[\.,:;'\"]+$": "",
    **{symbol: "" for symbol in LEGAL_SYMBOLS},
}
INCLUSION_SUPPRESSIONS = ["phase", "trial"]
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
    "cardiac",
    "polymers",
    "gene",
    "fatty",
    "abuse",
    "motion",
    "panic",
    "addiction",
    "mood",
    "regulatory",
    "region",
    "viral",
    "chronic",
    "joint",
    "digits",
    "protein",
    "complex",
    "death",
    "coding" "regulation",
    "mrna",
    "cell",
    "nervous",  # CNS
    "group",
    "plasma",
    "antibody",
    "dry",  # dry eye
]

DEFAULT_ADDITIONAL_COMMON_WORDS = [
    "(i)",  # so common in patents, e.g. "general formula (I)"
]


class EntityCleaner:
    """
    Class for cleaning entities

    Usage:
    ```
    import system; system.initialize()
    from common.ner.cleaning import EntityCleaner
    clean = EntityCleaner()
    clean(["atherosclerotic changes"])
    ```
    """

    def __init__(
        self,
        additional_common_words: list[str] = DEFAULT_ADDITIONAL_COMMON_WORDS,
        char_suppressions: dict[str, str] = CHAR_SUPPRESSIONS,
    ):
        self.additional_common_words = additional_common_words
        self.char_suppressions = char_suppressions
        self.__common_words = None
        self.nlp = Spacy.get_instance(disable=["ner"])

    @property
    def common_words(self) -> list[str]:
        """
        Get common words from a file + additional common words
        """
        if self.__common_words is None:
            with open("10000words.txt", "r") as file:
                vocab_words = file.read().splitlines()
            return [*vocab_words, *self.additional_common_words]
        return self.__common_words

    def filter_common_terms(
        self, entities: list[str], exception_list: list[str] = DEFAULT_EXCEPTION_LIST
    ) -> list[str]:
        """
        Filter out common terms from a list of entities, e.g. "vaccine candidates"

        Args:
            entities (list[T]): list of entities
            exception_list (list[str]): list of exceptions to the common terms
        """

        def __is_common(name):
            # remove punctuation and make lowercase
            words = [token.lemma_ for token in self.nlp(name)]

            # check if all words are in the vocab
            is_common = set(words).issubset(self.common_words)

            # check if any words are in the exception list
            is_excepted = bool(set(exception_list) & set(words))

            is_common_not_excepted = is_common and not is_excepted

            if is_common_not_excepted:
                logging.debug(f"Removing common term: {name}")
            elif is_excepted:
                logging.debug(f"Keeping exception term: {name}")
            return is_common_not_excepted

        def __is_uncommon(item):
            return not __is_common(item)

        new_list = [(ent if __is_uncommon(ent) else "") for ent in entities]
        return new_list

    def normalize_entity_names(self, entities: list[str]) -> list[str]:
        """
        Normalize entity name
        - remove certain characters
        - removes double+ spaces
        - lemmatize?

        Args:
            entities (list[T]): entities
        """

        def remove_chars(entity_name: str) -> str:
            for pattern, replacement in self.char_suppressions.items():
                entity_name = re.sub(pattern, replacement, entity_name)
            return entity_name

        def decode_html(entity_name: str) -> str:
            return html.unescape(entity_name)

        def remove_duplicative_phrasing(entity_name: str) -> str:
            duplicative_phrases = {
                "diseases and conditions": "diseases",
                "conditions and diseases": "diseases",
            }
            steps = [
                lambda s: re.sub(rf"\b{dup}\b", canonical, s)
                for dup, canonical in duplicative_phrases.items()
            ]
            text = reduce(lambda x, func: func(x), steps, entity_name)
            return text

        def normalize_entity(entity: str) -> str:
            text = entity[0] if isinstance(entity, DocEntity) else entity
            cleaning_steps = [
                decode_html,
                remove_chars,
                remove_extra_spaces,
                remove_duplicative_phrasing,
                rearrange_terms,
                lemmatize_tail,
                normalize_by_pos,
                lambda s: s.lower(),
            ]
            normalized = reduce(lambda x, func: func(x), cleaning_steps, text)
            return normalized

        return [normalize_entity(entity) for entity in entities]

    def suppress(self, entities: list[str]) -> list[str]:
        """
        Filter out irrelevant entities

        Args:
            entities (list[str]): entities
        """

        def should_keep(entity) -> bool:
            name = entity[0] if isinstance(entity, tuple) else entity
            is_suppressed = any([sup in name.lower() for sup in INCLUSION_SUPPRESSIONS])
            return not is_suppressed

        return [(entity if should_keep(entity) else "") for entity in entities]

    @staticmethod
    def get_text(entity) -> str:
        return entity[0] if isinstance(entity, tuple) else entity

    @staticmethod
    def return_to_type(modified_texts: list[str], orig_ents: list[T]) -> list[T]:
        if len(modified_texts) != len(orig_ents):
            raise ValueError("Modified text must be same length as original entities")

        if is_entity_doc_list(orig_ents):
            doc_ents = [
                DocEntity(*orig_ents[i][0:4], modified_texts[i], orig_ents[i][5])
                for i in range(len(orig_ents))
            ]
            return cast(list[T], doc_ents)

        if is_string_list(orig_ents):
            return cast(list[T], modified_texts)

        raise ValueError("Original entities must be a list of strings or DocEntities")

    def clean(
        self,
        entities: list[T],
        filter_exception_list: list[str] = DEFAULT_EXCEPTION_LIST,
    ) -> list[T]:
        """
        Sanitize entity list
        - filters out (some) excessively general entities
        - dedups
        - normalizes & lemmatizes entity names

        Args:
            entities (list[T]): entities
        """
        if not isinstance(entities, list):
            raise ValueError("Entities must be a list")

        cleaning_steps: list[CleanFunction] = [
            self.suppress,
            self.normalize_entity_names,
            partial(self.filter_common_terms, exception_list=filter_exception_list),
            dedup,
        ]

        string_ents = [self.get_text(ent) for ent in entities]
        sanitized = reduce(lambda x, func: func(x), cleaning_steps, string_ents)

        return self.return_to_type(sanitized, entities)

    def __call__(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
