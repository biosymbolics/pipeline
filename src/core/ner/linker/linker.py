"""
Term Normalizer
"""
import logging
from typing import Sequence
from pydash import compact


from ..types import CanonicalEntity, DocEntities, DocEntity

LinkedEntityMap = dict[str, CanonicalEntity]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MIN_SIMILARITY = 0.85


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

    def __init__(self, min_similarity: float = MIN_SIMILARITY):
        """
        Initialize term normalizer using existing model
        """
        # lazy (Umls is big)
        logger.info("Loading CompositeCandidateSelector (slow...)")
        from .composite_candidate_selector import CompositeCandidateSelector

        self.candidate_generator = CompositeCandidateSelector(
            min_similarity=min_similarity
        )

    def link(self, entity_set: Sequence[DocEntity]) -> list[CanonicalEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (Sequence[str]): list of terms to normalize
        """
        if len(entity_set) == 0:
            logging.warning("No entities to link")
            return []

        linked_entities = compact([self.candidate_generator(e) for e in entity_set])

        logging.info("Completed linking batch of %s entity sets", len(entity_set))

        return linked_entities

    def __call__(self, entity_set: Sequence[DocEntity]) -> list[CanonicalEntity]:
        return self.link(entity_set)
