"""
Term Normalizer
"""
from typing import Union
from scispacy.candidate_generation import (
    CandidateGenerator,
    UmlsKnowledgeBase,
    MentionCandidate,
)
from scispacy.linking_utils import Entity as SpacyEntity

MIN_SIMILARITY = 0.85


class TermNormalizer:
    """
    TermNormalizer

    Example:
        >>> normalizer = TermNormalizer()
        >>> normalizer(["APP", "haemorrage", "MEGFII", "anaemia"])
        ('APP gene', 'haemorrage', 'MEGFII', 'Anemia')
    """

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        self.candidate_generator = CandidateGenerator(name="umls")
        self.kb = UmlsKnowledgeBase()

    def __get_normalized_entity(
        self, candidate: MentionCandidate
    ) -> Union[dict[str, SpacyEntity], None]:
        """
        Get normalized name from candidate if suggestions exceed min similarity
        """
        if len(candidate) > 0 and candidate[0].similarities[0] > MIN_SIMILARITY:
            canonical_name = self.kb.cui_to_entity[candidate[0].concept_id]
            return canonical_name

        return None

    def generate_map(self, terms: list[str]) -> dict[str, SpacyEntity]:
        """
        Generate a map of terms to normalized/canonical entities (containing name and id)

        Args:
            terms (list[str]): list of terms to normalize

        Returns:
            dict[str, SpacyEntity]: mapping of terms to canonical entities
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
        return [term_map["term"]["canonical_name"] or term for term in terms]
