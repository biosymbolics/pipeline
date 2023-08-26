"""
Term Normalizer
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Union

# import torch
from scispacy.candidate_generation import MentionCandidate

from .types import KbLinker, CanonicalEntity

LinkedEntityMap = dict[str, CanonicalEntity]

MIN_SIMILARITY = 0.85
ONTOLOGY = "umls"


class TermLinker:
    """
    TermLinker

    - associates UMLS canonical entity if found
    - adds synonyms to synonym store
    - if no canonical entry found, then closest synonym is used

    Example:
        >>> linker = TermLinker()
        >>> linker(["APP", "haemorrage", "MEGFII", "anaemia"])
    """

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        # torch.device("mps")  # does this work?

        # lazy (Umls is big)
        from scispacy.candidate_generation import (
            CandidateGenerator,
            UmlsKnowledgeBase,
        )

        self.candidate_generator = CandidateGenerator()
        self.kb: KbLinker = UmlsKnowledgeBase()  # type: ignore

    def __get_canonical_entity(
        self, candidates: list[MentionCandidate]
    ) -> Union[CanonicalEntity, None]:
        """
        Get canonical candidate if suggestions exceed min similarity

        Args:
            candidates (list[MentionCandidate]): candidates
        """
        if len(candidates) > 0 and candidates[0].similarities[0] > MIN_SIMILARITY:
            entity = self.kb.cui_to_entity[candidates[0].concept_id]
            linked_entity = CanonicalEntity(
                entity.concept_id,
                entity.canonical_name,
                entity.aliases,
            )
            return linked_entity

        return None

    def generate_map(self, terms: list[str]) -> LinkedEntityMap:
        """
        Generate a map of terms to normalized/canonical entities (containing name and id)

        Args:
            terms (list[str]): list of terms to normalize

        Returns:
            LinkedEntityMap: mapping of terms to canonical entities
        """
        logging.info("Starting candidate generation")
        start_time = time.time()
        candidates = self.candidate_generator(terms, 1)
        logging.info(
            "Finished generating candidates (took %s)", time.time() - start_time
        )
        canonical_entities = [self.__get_canonical_entity(c) for c in candidates]
        entity_map = dict(zip(terms, canonical_entities))
        return {
            key: value
            for key, value in entity_map.items()
            if value is not None and len(value) > 1 and len(key) > 1
        }

    def link(self, terms: list[str]) -> list[tuple[str, CanonicalEntity | None]]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (list[str]): list of terms to normalize
        """
        if len(terms) == 0:
            logging.warning("No terms to link")
            return []

        canonical_map = self.generate_map(terms)

        # TODO: what is parallelism offering here??
        with ThreadPoolExecutor(max_workers=4) as executor:
            linked_entities = list(executor.map(lambda e: canonical_map.get(e), terms))
            logging.info("Completed linking batch of %s terms", len(terms))

        executor.shutdown()

        # should always be the same length ("" for omitted terms)
        assert len(terms) == len(linked_entities)

        return list(zip(terms, linked_entities))

    def __call__(self, terms: list[str]) -> list[tuple[str, CanonicalEntity | None]]:
        return self.link(terms)
