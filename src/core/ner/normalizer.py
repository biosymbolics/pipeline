"""
Linking/cleaning of terms
"""
from typing import Sequence

from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker.linker import TermLinker
from core.ner.linker.types import CandidateSelectorType
from core.ner.spacy import get_transformer_nlp
from core.ner.types import DocEntity


class TermNormalizer:
    """
    Normalizes and attempts to link entities.
    If no canonical entity found, then normalized term is returned.

    Usage:
        normalizer = TermNormalizer()
        terms = normalizer.normalize([
            "Tipranavir (TPV)",
            "BILR 355 - D4",
            "bivatuzumab mertansine",
            "BIBT 986 BS - single rising dose",
            "RDEA3170 10 mg",
        ])
        [(t[0], t[1].name) for t in terms]
    """

    def __init__(
        self,
        link: bool = True,
        candidate_selector: CandidateSelectorType | None = None,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        """
        Initialize term normalizer
        """
        if link:
            self.term_linker: TermLinker | None = (
                TermLinker(candidate_selector) if candidate_selector else TermLinker()
            )
        else:
            self.term_linker = None

        self.cleaner = EntityCleaner(
            additional_cleaners=additional_cleaners,
        )
        self.candidate_selector = candidate_selector

        if self.candidate_selector == "CandidateSelector":
            self.nlp = None
        else:
            # used for vectorization if direct entity linking
            self.nlp = get_transformer_nlp()

    def normalize(self, entity_sets: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
        """
        # removed_suppressed must be false to properly index against original terms
        cleaned_entity_sets = self.cleaner.clean(entity_sets, remove_suppressed=False)

        if self.term_linker is not None:
            return self.term_linker.link(cleaned_entity_sets)

        return cleaned_entity_sets

    def normalize_strings(self, terms: Sequence[str]) -> list[DocEntity]:
        if self.candidate_selector != "CandidateSelector":
            if self.nlp is None:
                raise ValueError(
                    "nlp model required for vectorization-based candidate selection"
                )
            docs = self.nlp.pipe(terms)
        else:
            docs = [None for _ in terms]

        doc_ents = [
            DocEntity(
                term=term,
                type="unknown",
                start_char=0,
                end_char=0,
                normalized_term=term,
                spacy_doc=doc,  # required for comparing semantic similarity of potential matches
            )
            for term, doc in zip(terms, docs)
        ]
        return self.normalize(doc_ents)

    def __call__(self, *args, **kwargs) -> list[DocEntity]:
        return self.normalize(*args, **kwargs)
