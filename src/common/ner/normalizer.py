"""
Term Normalizer
"""
from scispacy.candidate_generation import (
    CandidateGenerator,
    UmlsKnowledgeBase,
    MentionCandidate,
)


class TermNormalizer:
    """
    TermNormalizer

    Example:
        >>> normalizer = TermNormalizer()
        >>> normalizer(["APP", "haemorrage", "MEGFII", "anaemia"])
        {'APP': 'APP gene', 'haemorrage': None, 'MEGFII': None, 'anaemia': 'Anemia'}
    """

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        self.candidate_generator = CandidateGenerator(name="umls")
        self.kb = UmlsKnowledgeBase()

    def __get_normalized_name(self, candidate: MentionCandidate):
        """
        Get normalized name from candidate
        """
        if len(candidate) > 0 and candidate[0].similarities[0] > 0.85:
            canonical_name = self.kb.cui_to_entity[
                candidate[0].concept_id
            ].canonical_name
            return canonical_name

        return None

    def __normalize(self, terms: list[str]) -> dict[str, str]:
        """
        Normalize a list of terms

        Args:
            terms (list[str]): list of terms to normalize

        Returns:
            dict[str, str]: mapping of terms to normalized names
        """
        candidates = self.candidate_generator(terms, 1)
        canonical_names = [self.__get_normalized_name(c) for c in candidates]
        result_map = dict(zip(terms, canonical_names))
        return result_map

    def __call__(self, terms: list[str]) -> dict[str, str]:
        return self.__normalize(terms)
