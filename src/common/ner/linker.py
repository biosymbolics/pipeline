"""
Term Normalizer
"""
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Union
from scispacy.candidate_generation import (
    CandidateGenerator,
    UmlsKnowledgeBase,
    MentionCandidate,
)
import torch

from common.ner.synonyms import SynonymStore


from .types import KbLinker, CanonicalEntity

MIN_SIMILARITY = 0.85
ONTOLOGY = "umls"

LinkedEntityMap = dict[str, CanonicalEntity]


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
        torch.device("mps")  # does this work?
        self.candidate_generator = CandidateGenerator()
        self.kb: KbLinker = UmlsKnowledgeBase()  # type: ignore
        self.synonym_store = SynonymStore()

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
        candidates = self.candidate_generator(terms, 1)
        logging.info("Finished generating candidates")
        canonical_entities = [self.__get_canonical_entity(c) for c in candidates]
        entity_map = dict(zip(terms, canonical_entities))
        return {
            key: value
            for key, value in entity_map.items()
            if value is not None and len(value) > 1 and len(key) > 1
        }

    def link(self, terms: list[str]) -> list[tuple[str, CanonicalEntity]]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (list[str]): list of terms to normalize
        """
        canonical_map = self.generate_map(terms)

        def link_entity(name: str) -> CanonicalEntity:
            # get canonical entity if exists
            canonical = canonical_map.get(name)
            canonical_id = canonical.id if canonical else None

            # create synonym regardless
            syn_doc = self.synonym_store.add_synonym(
                name,
                canonical_id,
                {"canonical_name": canonical.name if canonical else None},
            )

            # return canonical, otherwise pseudo-canonical from synonym
            return canonical or CanonicalEntity(
                syn_doc["canonical_id"],
                syn_doc.get("metadata", {}).get("canonical_name") or "",
                [],
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            linked_entities = list(executor.map(link_entity, terms))
            logging.info("Completed linking batch of %s terms", len(terms))

        executor.shutdown()

        return list(zip(terms, linked_entities))

    def __call__(self, terms: list[str]) -> list[tuple[str, CanonicalEntity]]:
        return self.link(terms)
