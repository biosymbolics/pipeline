from typing import Sequence
from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
)
import logging

from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides

from .types import AbstractCandidateSelector, EntityWithScore
from .utils import candidate_to_canonical, apply_umls_word_overrides, score_candidate

MIN_SIMILARITY = 0.85
UMLS_KB = None
K = 10


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateSelector(AbstractCandidateSelector):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    Returns the candidate with the highest similarity score
    """

    @overrides(AbstractCandidateSelector)
    def __init__(self, *args, min_similarity: float = MIN_SIMILARITY, **kwargs):
        global UMLS_KB

        if UMLS_KB is None:
            from scispacy.linking_utils import UmlsKnowledgeBase

            UMLS_KB = UmlsKnowledgeBase()

        self.kb = UMLS_KB
        self.candidate_generator = CandidateGenerator(*args, kb=UMLS_KB, **kwargs)

        if min_similarity > 1:
            raise ValueError("min_similarity must be <= 1")
        elif min_similarity > 0.85:
            logger.warning(
                "min_similarity is high, this may result in many missed matches"
            )

        self.min_similarity = min_similarity

    def _score_candidate(
        self,
        concept_id: str,
        matching_aliases: Sequence[str],
        similarity: float,
    ) -> float:
        return score_candidate(
            concept_id,
            self.kb.cui_to_entity[concept_id].canonical_name,
            self.kb.cui_to_entity[concept_id].types,
            self.kb.cui_to_entity[concept_id].aliases,
            matching_aliases=matching_aliases,
            syntactic_similarity=similarity,
        )

    def _get_best_candidate(self, text: str) -> tuple[MentionCandidate, float] | None:
        """
        Select the best candidate for a mention
        """
        _candidates = self.candidate_generator([text], k=K)[0]
        if len(_candidates) == 0:
            logger.warning(f"No candidates found for {text}")
            return None

        # apply word overrides (e.g. if term is "modulator", give explicit UMLS match)
        # (because the best-fit UMLS term is "Biological Response Modifiers", which is low syntactic similarity)
        candidates = apply_umls_word_overrides(text, _candidates)

        # filter out candidates with low similarity
        sufficiently_similiar_candidates = [
            candidate
            for candidate in candidates
            if candidate.similarities[0] >= self.min_similarity
        ]

        if len(sufficiently_similiar_candidates) == 0:
            logger.debug(
                f"No candidates found for {text} with similarity >= {self.min_similarity}"
            )
            # tfidf vectorizer and/or the UMLS data have inconsistent handling of hyphens
            if "-" in text:
                # try replacing - with empty string
                res = self._get_best_candidate(text.replace("-", ""))
                if res is None:
                    # try replacing - with space (which will result in confusion if it separates out single chars)
                    return self._get_best_candidate(text.replace("-", " "))
                return res
            return None

        # score candidates
        scored_candidates = [
            (
                candidate,
                self._score_candidate(
                    candidate.concept_id,
                    candidate.aliases,
                    candidate.similarities[0],
                ),
            )
            for candidate in sufficiently_similiar_candidates
        ]
        # sort by score
        sorted_candidates = sorted(
            scored_candidates, key=lambda sc: sc[1], reverse=True
        )
        return sorted_candidates[0]

    @overrides(AbstractCandidateSelector)
    def select_candidate(self, text: str) -> EntityWithScore | None:
        """
        Select the best candidate for a mention
        """
        res = self._get_best_candidate(text)

        if res is None:
            return None

        candidate, score = res
        top_canonical = candidate_to_canonical(candidate, self.kb)
        return top_canonical, score

    @overrides(AbstractCandidateSelector)
    def select_candidate_from_entity(self, entity: DocEntity) -> EntityWithScore | None:
        return self.select_candidate(entity.term)

    @overrides(AbstractCandidateSelector)
    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidate = self.select_candidate(entity.term)

        if candidate is None:
            return None

        return candidate[0]
