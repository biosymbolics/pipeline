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

from common.ner.cleaning import EntityCleaner

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
        torch.device("mps")  # does this work?
        self.candidate_generator = CandidateGenerator()
        self.kb: KbLinker = UmlsKnowledgeBase()  # type: ignore
        # self.synonym_store = SynonymStore()

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

    def link(self, terms: list[str]) -> list[tuple[str, CanonicalEntity | None]]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (list[str]): list of terms to normalize
        """
        canonical_map = self.generate_map(terms)

        with ThreadPoolExecutor(max_workers=4) as executor:
            linked_entities = list(executor.map(lambda e: canonical_map.get(e), terms))
            logging.info("Completed linking batch of %s terms", len(terms))

        executor.shutdown()

        assert len(terms) == len(
            linked_entities
        )  # should always be the same length ("" for omitted terms)

        return list(zip(terms, linked_entities))

    def __call__(self, terms: list[str]) -> list[tuple[str, CanonicalEntity | None]]:
        return self.link(terms)


class TermNormalizer:
    """
    Normalizes and attempts to link entities.
    If no canonical entity found, then normalized term is returned
    Note that original term should always be preserved in order to keep association to original source.
    """

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        self.term_linker: TermLinker = TermLinker()
        self.cleaner: EntityCleaner = EntityCleaner()

    def normalize(self, terms: list[str]) -> list[tuple[str, CanonicalEntity]]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (list[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
            - will return fewer terms than input, if term meets conditions for suppression
        """
        normalized = self.cleaner.clean(terms)
        links = self.term_linker.link(normalized)

        def get_canonical(
            entity: tuple[str, CanonicalEntity | None], normalized: str
        ) -> CanonicalEntity:
            if entity[1] is None:
                return CanonicalEntity(id="", name=normalized, aliases=[])
            return entity[1]

        tups = [
            (t[0], get_canonical(t[1], t[2]))
            for t in zip(terms, links, normalized)
            if len(t[0]) > 0
        ]
        return tups

    def __call__(self, *args, **kwargs) -> list[tuple[str, CanonicalEntity]]:
        return self.normalize(*args, **kwargs)
