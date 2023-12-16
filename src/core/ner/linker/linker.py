"""
Term Normalizer
"""
import logging
from typing import Sequence


from ..types import CanonicalEntity, DocEntity

LinkedEntityMap = dict[str, CanonicalEntity]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        # lazy (Umls is big)
        logger.info("Loading CompositeCandidateSelector (slow...)")
        from .composite_candidate_selector import CompositeCandidateSelector

        self.candidate_generator = CompositeCandidateSelector()

    def link(self, entity_set: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (Sequence[str]): list of terms to normalize
        """
        if len(entity_set) == 0:
            logging.warning("No entities to link")
            return []

        # generate the candidates (somewhat time consuming)
        canonical_entities = [self.candidate_generator(e) for e in entity_set]

        def get_canonical(ce: CanonicalEntity | None, de: DocEntity) -> CanonicalEntity:
            # create a pseudo-canonical entity if no canonical entity found
            if ce is None:
                return CanonicalEntity(
                    id="", name=de.normalized_term or de.term, aliases=[de.term]
                )
            return ce

        linked_doc_ents = [
            DocEntity(**{**e, "linked_entity": get_canonical(ce[0], e)})
            for e, ce in zip(entity_set, canonical_entities)
        ]

        logging.info("Completed linking batch of %s entity sets", len(entity_set))

        return linked_doc_ents

    def __call__(self, entity_set: Sequence[DocEntity]) -> list[DocEntity]:
        return self.link(entity_set)
