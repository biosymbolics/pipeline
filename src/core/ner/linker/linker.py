"""
Term Normalizer
"""
import logging
import time
from typing import Sequence


from ..types import CanonicalEntity

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

    def generate_map(self, terms: Sequence[str]) -> LinkedEntityMap:
        """
        Generate a map of terms to normalized/canonical entities (containing name and id)

        Args:
            terms (Sequence[str]): list of terms to normalize

        Returns:
            LinkedEntityMap: mapping of terms to canonical entities
        """
        logging.info("Starting candidate generation")
        start_time = time.time()
        entities = self.candidate_generator(list(terms))

        # INFO:root:Finished candidate generation (took 1703 seconds)
        logging.info(
            "Finished candidate generation (took %s seconds)",
            round(time.time() - start_time),
        )
        return {
            key: value
            for key, value in zip(terms, entities)
            if value is not None and len(key) > 1
        }

    def link(
        self, terms: Sequence[str]
    ) -> Sequence[tuple[str, CanonicalEntity | None]]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (Sequence[str]): list of terms to normalize
        """
        if len(terms) == 0:
            logging.warning("No terms to link")
            return []

        canonical_map = self.generate_map(terms)
        linked_entities = [(t, canonical_map.get(t)) for t in terms]

        logging.info("Completed linking batch of %s terms", len(terms))

        return linked_entities

    def __call__(
        self, terms: Sequence[str]
    ) -> Sequence[tuple[str, CanonicalEntity | None]]:
        return self.link(terms)
