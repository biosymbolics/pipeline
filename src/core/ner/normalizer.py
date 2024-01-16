"""
Linking/cleaning of terms
"""
from typing import Sequence
import logging

from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker.linker import TermLinker
from core.ner.linker.types import CandidateSelectorType
from core.ner.spacy import get_transformer_nlp
from core.ner.types import DocEntity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def normalize(self, doc_entities: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
        """
        # removed_suppressed must be false to properly index against original terms
        cleaned_entities = self.cleaner.clean(doc_entities, remove_suppressed=False)

        if self.term_linker is not None:
            return self.term_linker.link(cleaned_entities)

        return cleaned_entities

    def normalize_strings(self, terms: Sequence[str]) -> list[DocEntity]:
        # uniqify since all we have is terms and no context/doc
        # (therefore identical terms are identical entities)
        uniq_terms = list(set(terms))

        if self.candidate_selector not in [
            "CandidateSelector",
            "CompositeCandidateSelector",
        ]:
            if self.nlp is None:
                raise ValueError(
                    "nlp model required for vectorization-based candidate selection"
                )
            docs = self.nlp.pipe(uniq_terms)
        else:
            logger.info("No nlp model required for CandidateSelector")
            docs = [None for _ in uniq_terms]

        doc_ents = [
            DocEntity(
                term=term,
                type="unknown",
                start_char=0,
                end_char=0,
                normalized_term=term,
                spacy_doc=doc,  # required for comparing semantic similarity of potential matches
            )
            for term, doc in zip(uniq_terms, docs)
        ]

        logger.info("Normalizing %s terms (%s non-unique)", len(doc_ents), len(terms))
        normalized = self.normalize(doc_ents)
        normalized_map = {n.term: n for n in normalized}
        return [normalized_map[t] for t in terms]

    def __call__(self, *args, **kwargs) -> list[DocEntity]:
        return self.normalize(*args, **kwargs)
