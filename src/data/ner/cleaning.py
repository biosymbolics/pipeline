"""
Utils for the NER pipeline
"""
from abc import abstractmethod
import time
from typing import Iterable, TypeVar, Union, cast
import re
from functools import partial, reduce
import logging
import html
from typing_extensions import Protocol

from utils.re import remove_extra_spaces, LEGAL_SYMBOLS
from typings.core import is_string_list

from .spacy import Spacy
from .types import DocEntity, is_entity_doc_list
from .utils import lemmatize_tails, normalize_by_pos, rearrange_terms

T = TypeVar("T", bound=Union[DocEntity, str])


class CleanFunction(Protocol):
    @abstractmethod
    def __call__(self, terms: list[str], n_process: int) -> list[str]:
        pass


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
    "(1)",
    "sec",
    "second",
]

MAX_N_PROCESS = 4


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
        additional_common_words: list[str] = DEFAULT_ADDITIONAL_COMMON_WORDS,
        char_suppressions: dict[str, str] = CHAR_SUPPRESSIONS,
        parallelize: bool = True,
    ):
        self.additional_common_words = additional_common_words
        self.char_suppressions = char_suppressions
        self.parallelize = parallelize
        self.__common_words = None
        self.nlp = Spacy.get_instance(disable=["ner"])._nlp

    def get_n_process(self, num_entries: int) -> int:
        """
        Get the number of processes to use for parallelization in nlp pipeline

        only parallelize if
            1) parallelize is set to true and
            2) there are more than 800 entities (otherwise the overhead probably exceeds the benefits)
        """
        parallelize = self.parallelize and num_entries > 800
        return MAX_N_PROCESS if parallelize else 1

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
        self,
        terms: list[str],
        exception_list: list[str] = DEFAULT_EXCEPTION_LIST,
        n_process: int = 1,
    ) -> list[str]:
        """
        Filter out common terms from a list of terms, e.g. "vaccine candidates"

        Args:
            terms (list[str]): list of terms
            exception_list (list[str]): list of exceptions to the common terms
        """
        # doing in bulk for perf
        token_sets = self.nlp.pipe(terms, n_process=n_process)
        term_lemmas = [
            (term, set([t.lemma_ for t in tokens]))
            for term, tokens in zip(terms, token_sets)
        ]

        def __is_common(term: str, lemmas: set[str]) -> bool:
            # check if all words are in the vocab
            is_common = lemmas.issubset(self.common_words)

            # check if any words are in the exception list
            is_excepted = bool(set(exception_list) & set(lemmas))

            is_common_not_excepted = is_common and not is_excepted

            if is_common_not_excepted:
                logging.debug(f"Removing common term: {term}")
            elif is_excepted:
                logging.debug(f"Keeping exception term: {term}")
            return is_common_not_excepted

        def __is_uncommon(term: str, lemmas: set[str]) -> bool:
            return not __is_common(term, lemmas)

        new_list = list([(tw[0] if __is_uncommon(*tw) else "") for tw in term_lemmas])
        return new_list

    def normalize_terms(self, terms: list[str], n_process: int = 1) -> list[str]:
        """
        Normalize terms
        - remove certain characters
        - removes double+ spaces
        - lemmatize?

        Args:
            terms (list[str]): terms
            n_process (int): number of processes to use for parallelization
        """
        start = time.time()

        def remove_chars(_terms: list[str]) -> Iterable[str]:
            def __remove_chars(term):
                for pattern, replacement in self.char_suppressions.items():
                    term = re.sub(pattern, replacement, term)
                return term

            for term in _terms:
                yield __remove_chars(term)

        def decode_html(_terms: list[str]) -> Iterable[str]:
            for term in _terms:
                yield html.unescape(term)

        def lower(_terms: list[str]) -> Iterable[str]:
            for term in _terms:
                yield term.lower()

        def unwrap(_terms: list[str]) -> Iterable[str]:
            for term in _terms:
                yield term.strip("()[]")

        def normalize_phrasing(_terms: list[str]) -> Iterable[str]:
            phrases = {
                "diseases and conditions": "diseases",
                "conditions and diseases": "diseases",
                "diseases and disorders": "diseases",
                "disorders and diseases": "diseases",
                "analogues?": "analog",
                "drug delivery": "delivery",
                "tumours?": "tumor",
                "receptor agonists?": "agonist",  # ??
                "receptor antagonists?": "antagonist",  # ??
                "receptor modulators?": "modulator",
                "activity modulators?": "modulator",
                "binding modulators?": "modulator",
                "inhibitor compounds?": "inhibitor",
                "activity modulators?": "modulator",
                "small molecule inhibitors?": "inhibitor",
                "associated proteins?": "protein",
                "transporter inhibitors?": "transport inhibitor",
                # "disease states mediated by": "associated disease", # disease states mediated by CCR5 (rearrange)
                "mediated conditions?": "associated disease",
                "mediated diseases?": "associated disease",
                "related conditions?": "associated disease",
                "related diseases?": "associated disease",
                "antibodies?": "antibody",
                "diarrhoea": "diarrhea",
                "faecal": "fecal",
            }
            steps = [
                lambda s: re.sub(rf"\b{dup}\b", canonical, s)
                for dup, canonical in phrases.items()
            ]

            for term in _terms:
                yield reduce(lambda x, func: func(x), steps, term)

        def exec_func(func, x):
            logging.debug("Executing function: %s", func)
            return func(x)

        cleaning_steps = [
            decode_html,
            remove_chars,
            unwrap,
            remove_extra_spaces,
            normalize_phrasing,
            partial(rearrange_terms, n_process=n_process),
            partial(lemmatize_tails, n_process=n_process),
            partial(normalize_by_pos, n_process=n_process),
            lower,
        ]

        normalized = list(
            reduce(lambda x, func: exec_func(func, x), cleaning_steps, terms)
        )

        logging.debug(
            "Normalized %s terms in %s seconds",
            len(terms),
            round(time.time() - start, 2),
        )
        return normalized

    def __suppress(self, terms: list[str]) -> list[str]:
        """
        Filter out irrelevant terms

        Args:
            terms (list[str]): terms
        """

        def should_keep(term) -> bool:
            is_suppressed = any([sup in term.lower() for sup in INCLUSION_SUPPRESSIONS])
            return not is_suppressed

        return [(term if should_keep(term) else "") for term in terms]

    @staticmethod
    def __get_text(entity) -> str:
        return entity[0] if isinstance(entity, tuple) else entity

    @staticmethod
    def __return_to_type(
        modified_texts: list[str], orig_ents: list[T], remove_suppressed: bool = False
    ) -> list[T]:
        if len(modified_texts) != len(orig_ents):
            logging.info(
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
        entities: list[T],
        common_exception_list: list[str] = DEFAULT_EXCEPTION_LIST,
        remove_suppressed: bool = False,
    ) -> list[T]:
        """
        Sanitize entity list
        - filters out (some) excessively general entities
        - normalizes & lemmatizes entity names

        Args:
            entities (list[T]): entities
            common_exception_list (list[str], optional): list of exceptions to the common terms. Defaults to DEFAULT_EXCEPTION_LIST.
            remove_suppressed (bool, optional): remove empties? Defaults to False (leaves empty spaces in, to maintain order)
        """
        start = time.time()
        if not isinstance(entities, list):
            raise ValueError("Entities must be a list")

        num_processes = self.get_n_process(len(entities))

        if num_processes > 1:
            logging.info(
                "Using %s processes for count %s", num_processes, len(entities)
            )

        cleaning_steps: list[CleanFunction] = [
            lambda terms, n_process: self.__suppress(terms),
            self.normalize_terms,
            partial(
                self.filter_common_terms,
                exception_list=common_exception_list,
            ),
        ]

        terms = [self.__get_text(ent) for ent in entities]
        cleaned = reduce(
            lambda x, func: func(x, n_process=num_processes), cleaning_steps, terms
        )

        logging.debug(
            "Cleaned %s entities in %s seconds",
            len(entities),
            round(time.time() - start, 2),
        )

        return self.__return_to_type(
            cleaned, entities, remove_suppressed=remove_suppressed
        )

    def __call__(self, *args, **kwargs):
        return self.clean(*args, **kwargs)
