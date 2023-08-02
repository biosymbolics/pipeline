"""
Utils for the NER pipeline
"""
from typing import Callable, TypeVar, Union, cast
import re
from functools import reduce
import logging
import html
from spacy.tokens import Doc

from clients.spacy import Spacy
from common.ner.utils import lemmatize_tail
from common.utils.list import dedup
from common.utils.re import remove_extra_spaces, LEGAL_SYMBOLS

from .types import DocEntity

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
        self, entities: list[T], exception_list: list[str] = DEFAULT_EXCEPTION_LIST
    ) -> list[T]:
        """
        Filter out common terms from a list of entities, e.g. "vaccine candidates"

        Args:
            entities (list[T]): list of entities
            exception_list (list[str]): list of exceptions to the common terms
        """

        def __is_common(item: T):
            name = item[0] if isinstance(item, tuple) else item

            # remove punctuation and make lowercase
            words = [token.lemma_ for token in self.nlp(name)]

            # check if all words are in the vocab
            is_common = set(words).issubset(self.common_words)

            # check if any words are in the exception list
            is_excepted = bool(set(exception_list) & set(words))

            is_common_not_excepted = is_common and not is_excepted

            if is_common_not_excepted:
                logging.debug(f"Removing common term: {item}")
            elif is_excepted:
                logging.debug(f"Keeping exception term: {item}")
            return is_common_not_excepted

        def __is_uncommon(item):
            return not __is_common(item)

        return list(filter(__is_uncommon, entities))

    def normalize_entity_names(self, entities: list[T]) -> list[T]:
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

        def normalize_by_pos(entity_name: str) -> str:
            """
            Normalizes entity by POS

            Dashes:
                Remove and replace with "" if Spacy considers PUNCT and followed by NUM:
                    - APoE-4 -> apoe4 (NOUN(PUNCT)NUM)
                    - HIV-1 -> hiv1 (NOUN(PUNCT)NUM)

                Remove and replace with **space** if Spacy considers it PUNCT or ADJ:
                - sodium channel-mediated diseases (NOUN(PUNCT)VERB)
                - neuronal hypo-kinetic disease (NOUN(PUNCT)ADJ) # TODO: better if ""
                - Loeys-Dietz syndrome (PROPN(PUNCT)NOUN)
                - sleep-wake cycles (NOUN(PUNCT)NOUN)
                - low-grade prostate cancer (ADJ(PUNCT)NOUN)
                - non-insulin dependent diabetes mellitus (ADJ(ADJ)NOUN)
                - T-cell lymphoblastic leukemia (NOUN(ADJ)NOUN)
                - T-cell (NOUN(PUNCT)NOUN)
                - MAGE-A3 gene (PROPN(PUNCT)NOUN) # TODO: better if ""
                - Bcr-Abl (NOUN(ADJ)ADJ) -> Bcr-Abl # TODO

                Keep if Spacy considers is a NOUN
                - HLA-C (NOUN(NOUN)NOUN) -> HLA-C
                - IL-6 (NOUN(NOUN)NUM) -> IL-6

            Other changes:
              - Alzheimer's disease -> Alzheimer disease
            """
            dashes = ["–", "-"]
            tokens = self.nlp(
                re.sub(r"[–-]", " - ", entity_name)
            )  # otherwise spacy will keep it as one token

            def clean_by_pos(t, next_t):
                # spacy only marks a token as SPACE if it is hanging out in a weird place
                if t.pos_ == "SPACE":
                    return ""
                if t.text == "'s" and t.pos_ == "PART":
                    # alzheimer's disease -> alzheimer disease
                    return ""
                if t.text == "-":
                    if t.pos_ == "ADJ":
                        return ""
                    if t.pos_ == "PUNCT":
                        if next_t is not None and next_t.pos_ == "NUM":
                            # ApoE-4 -> apoe4
                            return ""
                        return " "
                    else:
                        # pos_ == NOUN, PROPN, etc
                        return "-"

                if next_t is not None and next_t.text in dashes:
                    return t.text  # omitting pre-dash space

                return t.text_with_ws

            return "".join(
                [
                    clean_by_pos(t, tokens[i] if len(tokens) > i + 1 else None)
                    for i, t in enumerate(tokens)
                ]
            )

        def normalize_entity(entity: T) -> T:
            text = entity[0] if isinstance(entity, DocEntity) else entity
            cleaning_steps = [
                decode_html,
                remove_chars,
                remove_extra_spaces,
                lemmatize_tail,
                normalize_by_pos,
                lambda s: s.lower(),
            ]
            normalized = reduce(lambda x, func: func(x), cleaning_steps, text)

            if isinstance(entity, DocEntity):
                return cast(T, DocEntity(*entity[0:4], normalized, entity[5]))
            return cast(T, normalized)

        return [normalize_entity(entity) for entity in entities]

    def suppress(self, entities: list[T]) -> list[T]:
        """
        Filter out irrelevant entities

        Args:
            entities (list[T]): entities
        """

        def should_keep(entity: T) -> bool:
            name = entity[0] if isinstance(entity, tuple) else entity
            is_suppressed = any([sup in name.lower() for sup in INCLUSION_SUPPRESSIONS])
            return not is_suppressed

        return [entity for entity in entities if should_keep(entity)]

    def __call__(self, entities: list[T]) -> list[T]:
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
            self.filter_common_terms,
            dedup,
        ]

        sanitized = reduce(lambda x, func: func(x), cleaning_steps, entities)

        return sanitized
