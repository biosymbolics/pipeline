"""
Term Normalizer
"""
import logging
import time
from typing import Sequence

from .types import AbstractCandidateSelector, CandidateSelectorType

from ..types import CanonicalEntity, DocEntity

LinkedEntityMap = dict[str, CanonicalEntity]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CANDIDATE_SELECTOR_MODULE = "core.ner.linker"


class TermLinker:
    """
    TermLinker

    - associates UMLS canonical entity if found
    - adds synonyms to synonym store
    - if no canonical entry found, then closest synonym is used

    Example:
        >>> linker = TermLinker()
        >>> linker(["DNMT1", "DNMT1 inhibitor"])
        >>> linker(["DNMT1 protein synthase inhibitor"])
    """

    def __init__(
        self,
        candidate_selector_class: CandidateSelectorType = "SemanticCandidateSelector",
        *args,
        **kwargs,
    ):
        """
        Initialize term normalizer using existing model
        """
        # lazy (UMLS is large)
        logger.info("Loading %s (might be slow...)", candidate_selector_class)
        modules = __import__(
            CANDIDATE_SELECTOR_MODULE, fromlist=[candidate_selector_class]
        )
        self.candidate_selector: AbstractCandidateSelector = getattr(
            modules, candidate_selector_class
        )(*args, **kwargs)

    def link(self, entity_set: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (Sequence[str]): list of terms to normalize
        """
        if len(entity_set) == 0:
            logging.warning("No entities to link")
            return []

        start = time.monotonic()

        # generate the candidates (somewhat time consuming)
        canonical_entities = [self.candidate_selector(e) for e in entity_set]

        def get_canonical(ce: CanonicalEntity | None, de: DocEntity) -> CanonicalEntity:
            # create a pseudo-canonical entity if no canonical entity found
            if ce is None:
                return CanonicalEntity(
                    id="", name=de.normalized_term or de.term, aliases=[de.term]
                )
            return ce

        linked_doc_ents = [
            DocEntity(**{**e, "canonical_entity": get_canonical(ce, e)})
            for e, ce in zip(entity_set, canonical_entities)
        ]

        logging.info(
            "Completed linking batch of %s entity sets, took %ss",
            len(entity_set),
            round(time.monotonic() - start),
        )

        return linked_doc_ents

    def __call__(self, entity_set: Sequence[DocEntity]) -> list[DocEntity]:
        return self.link(entity_set)
