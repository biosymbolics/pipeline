"""
Term Normalizer
"""
import logging
import time
from typing import List, NamedTuple, Sequence


from ..types import KbLinker, CanonicalEntity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MIN_SIMILARITY = 0.85
ONTOLOGY = "umls"

LinkedEntityMap = dict[str, CanonicalEntity]


# copy from scispacy to avoid the import
class MentionCandidate(NamedTuple):
    concept_id: str
    aliases: List[str]
    similarities: List[float]


class TermLinker:
    """
    TermLinker

    - associates UMLS canonical entity if found
    - adds synonyms to synonym store
    - if no canonical entry found, then closest synonym is used

    Example:
        >>> linker = TermLinker()
        >>> linker(["DNMT1", "DNMT1 inhibitor"])
    """

    def __init__(self, min_similarity: float = MIN_SIMILARITY):
        """
        Initialize term normalizer using existing model
        """
        # lazy (Umls is big)
        logger.info("Loading CompositeCandidateGenerator")
        from core.ner.linker.candidate_generator import (
            CompositeCandidateGenerator as CandidateGenerator,
        )

        self.candidate_generator = CandidateGenerator(min_similarity=min_similarity)
        self.kb: KbLinker = UmlsKnowledgeBase()  # type: ignore

    def __get_canonical_entity(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity

        Args:
            candidates (Sequence[MentionCandidate]): candidates
        """
        top_candidate = candidates[0] if len(candidates) > 0 else None

        if top_candidate is not None and top_candidate.similarities[0] < MIN_SIMILARITY:
            return None

        # go to kb to get canonical name
        entity = self.kb.cui_to_entity.get(candidates[0].concept_id)

        # if no entity, which will happen if we got a composite candidate:
        if entity is None:
            return CanonicalEntity(
                candidates[0].concept_id,
                candidates[0].aliases[0],
                candidates[0].aliases,
            )

        return CanonicalEntity(
            entity.concept_id if entity else candidates[0].concept_id,
            entity.canonical_name if entity else candidates[0].aliases[0],
            entity.aliases if entity else candidates[0].aliases,
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
        candidates = self.candidate_generator(list(terms), 1)
        logging.info(
            "Finished generating candidates (took %s)", time.time() - start_time
        )
        canonical_entities = [self.__get_canonical_entity(c) for c in candidates]  # type: ignore
        entity_map = dict(zip(terms, canonical_entities))
        return {
            key: value
            for key, value in entity_map.items()
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

        linked_entities = [canonical_map.get(t) for t in terms]
        logging.info("Completed linking batch of %s terms", len(terms))

        # should always be the same length ("" for omitted terms)
        assert len(terms) == len(linked_entities)

        return list(zip(terms, linked_entities))

    def __call__(
        self, terms: Sequence[str]
    ) -> Sequence[tuple[str, CanonicalEntity | None]]:
        return self.link(terms)
