"""
Utils for the NER pipeline
"""
from abc import abstractmethod
import time
from typing import Iterable, Mapping, Sequence, TypeVar, Union, cast
import regex as re
from functools import partial, reduce
import logging
import html
from typing_extensions import Protocol

from constants.patterns.intervention import (
    INTERVENTION_PREFIXES_GENERIC_RE,
    PRIMARY_MECHANISM_BASE_TERMS,
)
from constants.patterns.iupac import is_iupac
from data.common.biomedical.constants import PHRASE_REWRITES
from utils.re import get_or_re, remove_extra_spaces, LEGAL_SYMBOLS, RE_STANDARD_FLAGS
from typings.core import is_string_list

from .types import DocEntity, is_entity_doc_list
from .utils import depluralize_tails, normalize_by_pos, rearrange_terms

T = TypeVar("T", bound=Union[DocEntity, str])

RE_FLAGS = RE_STANDARD_FLAGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CleanFunction(Protocol):
    @abstractmethod
    def __call__(self, terms: Sequence[str]) -> Sequence[str]:
        pass


SUBSTITUTIONS = {
    r"\n": " ",
    "/": " ",
    r"[.,:;'\"]+$": "",  # trailing punct
    r"^[.,:;'\"]+": "",  # leading punct
    **{symbol: "" for symbol in LEGAL_SYMBOLS},
    INTERVENTION_PREFIXES_GENERIC_RE: " ",
}


class EntityCleaner:
    """
    Class for cleaning entities

    Usage:
    ```
    import system; system.initialize()
    from ner.cleaning import EntityCleaner
    clean = EntityCleaner()
    clean(["atherosclerotic changes"])
    ```
    """

    def __init__(
        self,
        substitutions: Mapping[str, str] = SUBSTITUTIONS,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        self.substitutions = substitutions
        self.additional_cleaners = additional_cleaners
        self.__removal_words: list[str] | None = None

    @property
    def removal_words(self) -> list[str]:
        """
        Get common words from a file + additional common words

        LEGACY
        """
        if self.__removal_words is None:
            with open("10000words.txt", "r") as file:
                vocab_words = file.read().splitlines()
                self.__removal_words = vocab_words

        return self.__removal_words

    def normalize_terms(self, terms: Sequence[str]) -> Sequence[str]:
        """
        Normalize terms
        - remove certain characters
        - removes double+ spaces
        - lemmatize?

        Args:
            terms (list[str]): terms
        """

        def make_substitutions(_terms: Sequence[str]) -> Iterable[str]:
            def __substitute_strings(term):
                for pattern, replacement in self.substitutions.items():
                    term = re.sub(pattern, replacement, term, flags=RE_FLAGS)

                return term

            for term in _terms:
                yield __substitute_strings(term)

        def decode_html(_terms: Sequence[str]) -> Iterable[str]:
            for term in _terms:
                yield html.unescape(term)

        def remove_after_newline(_terms: Sequence[str]) -> Iterable[str]:
            """
            Remove everything after a newline
            e.g. "asthma\n is a disease" -> "asthma"
            """
            for term in _terms:
                yield re.sub(r"\n.*", "", term, flags=RE_FLAGS)

        def lower(_terms: Sequence[str]) -> Iterable[str]:
            for term in _terms:
                yield term.lower()

        def unwrap(_terms: Sequence[str]) -> Iterable[str]:
            for term in _terms:
                if term.startswith("(") and term.endswith(")"):
                    yield term.strip("()")
                    continue
                yield term

        def format_parentheticals(_terms: Sequence[str]) -> Iterable[str]:
            for term in _terms:
                # if iupac term, don't mess with its parens
                if is_iupac(term):
                    yield term
                    continue

                # removes `(IL-2)` from `Interleukin-2 (IL-2) inhibitor`
                no_parenth = re.sub(
                    r"(?<=[ ,])(\([a-z-0-9 ]+\))(?=(?: |,|$))",
                    "",
                    term,
                    flags=RE_FLAGS,
                )
                # `poly(isoprene)` -> `polyisoprene``
                no_parens = re.sub(
                    r"\(([a-z-0-9]+)\)",
                    r"\1",
                    no_parenth,
                    flags=RE_FLAGS,
                )
                yield no_parens

        def normalize_phrases(_terms: Sequence[str]) -> Iterable[str]:
            def _map(s, syn, canonical):
                return re.sub(rf"\b{syn}s?\b", canonical, s, flags=RE_FLAGS)

            steps = [
                partial(_map, syn=syn, canonical=canonical)
                for syn, canonical in PHRASE_REWRITES.items()
            ]

            pharse_to_norm_re = get_or_re(
                list(PHRASE_REWRITES.keys()), enforce_word_boundaries=True
            )

            for term in _terms:
                if (
                    re.search(f".*{pharse_to_norm_re}.*", term, flags=RE_FLAGS)
                    is not None
                ):
                    yield reduce(lambda x, func: func(x), steps, term)
                else:
                    yield term

        def exec_func(func, x, debug: bool = True):
            if debug:
                start = time.time()
                res = list(func(x))
                logger.info(
                    "Executing function %s took %s", func, round(time.time() - start, 2)
                )
                return res
            return func(x)

        cleaning_steps = [
            decode_html,
            remove_after_newline,  # order matters (this before unwrap etc)
            unwrap,
            format_parentheticals,  # order matters (run after unwrap)
            make_substitutions,  # order matters (after unwrap/format_parentheticals)
            remove_extra_spaces,
            partial(
                rearrange_terms, base_patterns=list(PRIMARY_MECHANISM_BASE_TERMS.keys())
            ),
            depluralize_tails,
            normalize_by_pos,
            normalize_phrases,  # order matters (after rearrange)
            *self.additional_cleaners,
            remove_extra_spaces,
            lower,
        ]

        normalized = list(
            reduce(lambda x, func: exec_func(func, x), cleaning_steps, terms)
        )

        return normalized

    @staticmethod
    def __get_text(entity) -> str:
        return entity[0] if isinstance(entity, tuple) else entity

    @staticmethod
    def __return_to_type(
        modified_texts: Sequence[str],
        orig_ents: Sequence[T],
        remove_suppressed: bool = False,
    ) -> Sequence[T]:
        if len(modified_texts) != len(orig_ents):
            logger.info(
                "Modified text: %s, original entities: %s", modified_texts, orig_ents
            )
            raise ValueError("Modified text must be same length as original entities")

        if is_entity_doc_list(orig_ents):
            doc_ents = [
                DocEntity(*orig_ents[i][0:4], modified_texts[i], orig_ents[i][5])
                for i in range(len(orig_ents))
            ]
            if remove_suppressed:
                doc_ents = [d for d in doc_ents if len(d.normalized_term) > 0]
            return cast(list[T], doc_ents)

        if is_string_list(orig_ents):
            if remove_suppressed:
                modified_texts = [t for t in modified_texts if len(t) > 0]
            return cast(list[T], modified_texts)

        raise ValueError("Original entities must be a list of strings or DocEntities")

    def clean(
        self,
        entities: Sequence[T],
        remove_suppressed: bool = False,
    ) -> Sequence[T]:
        """
        Sanitize entity list
        - filters out (some) excessively general entities
        - normalizes & lemmatizes entity names

        Args:
            entities (list[T]): entities
            remove_suppressed (bool, optional): remove empties? Defaults to False (leaves empty spaces in, to maintain order)
        """
        start = time.time()
        if not isinstance(entities, list):
            raise ValueError("Entities must be a list")

        cleaned = self.normalize_terms([self.__get_text(ent) for ent in entities])

        # INFO:core.ner.cleaning:Cleaned 2268406 entities in 1957.85 seconds
        logger.info(
            "Cleaned %s entities in %s seconds",
            len(entities),
            round(time.time() - start, 2),
        )

        return self.__return_to_type(
            cleaned, entities, remove_suppressed=remove_suppressed
        )

    def __call__(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
