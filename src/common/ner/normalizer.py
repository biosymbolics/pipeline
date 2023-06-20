"""
Term Normalizer
"""
import logging
from typing import Union
from scispacy.candidate_generation import (
    CandidateGenerator,
    UmlsKnowledgeBase,
    MentionCandidate,
)
from scispacy.linking_utils import Entity as SpacyEntity

MIN_SIMILARITY = 0.85

NormalizationMap = dict[str, SpacyEntity]

ONTOLOGY = "umls"


class TermNormalizer:
    """
    TermNormalizer - normalizes terms using scispacy's UMLS candidate generator

    Example:
        >>> normalizer = TermNormalizer()
        >>> normalizer(["APP", "haemorrage", "MEGFII", "anaemia"])
        ('APP gene', 'haemorrage', 'MEGFII', 'Anemia')
    """

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        self.candidate_generator = CandidateGenerator(name=ONTOLOGY)
        self.kb = UmlsKnowledgeBase()

    def __get_normalized_entity(
        self, candidates: list[MentionCandidate]
    ) -> Union[SpacyEntity, None]:
        """
        Get normalized name from candidate if suggestions exceed min similarity

        Args:
            candidate (MentionCandidate): candidate
        """
        if len(candidates) > 0 and candidates[0].similarities[0] > MIN_SIMILARITY:
            entity = self.kb.cui_to_entity[candidates[0].concept_id]
            return entity

        return None

    def generate_map(self, terms: list[str]) -> NormalizationMap:
        """
        Generate a map of terms to normalized/canonical entities (containing name and id)

        Args:
            terms (list[str]): list of terms to normalize

        Returns:
            NormalizationMap: mapping of terms to canonical entities
        """
        candidates = self.candidate_generator(terms, 1)
        canonical_entities = [self.__get_normalized_entity(c) for c in candidates]
        entity_map = dict(zip(terms, canonical_entities))
        return {key: value for key, value in entity_map.items() if value is not None}

    def __call__(self, terms: list[str]) -> list[str]:
        """
        Normalize a list of terms
        """
        term_map = self.generate_map(terms)
        return [
            getattr(term_map.get(term, ()), "canonical_name", term) for term in terms
        ]
