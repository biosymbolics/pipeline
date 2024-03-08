import logging

import torch

from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from typings.documents.common import MentionCandidate

from .candidate_generator import CandidateGenerator
from .types import AbstractCandidateSelector, EntityWithScore
from .utils import (
    apply_match_retry_rewrites,
    candidate_to_canonical,
    apply_umls_word_overrides,
    score_candidate,
)

MIN_SIMILARITY = 0.85
UMLS_KB = None
K = 5


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateSelector(AbstractCandidateSelector):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    Returns the candidate with the highest similarity score
    """

    @overrides(AbstractCandidateSelector)
    def __init__(self, min_similarity: float = MIN_SIMILARITY):
        if min_similarity > 1:
            raise ValueError("min_similarity must be <= 1")
        elif min_similarity > 0.85:
            logger.warning(
                "min_similarity is high, this may result in many missed matches"
            )

        self.min_similarity = min_similarity
        self.candidate_generator = CandidateGenerator()

    async def _get_best_candidate(
        self, text: str, is_composite: bool
    ) -> tuple[MentionCandidate, float] | None:
        """
        Select the best candidate for a mention
        """
        vanilla_candidates = (await self.candidate_generator([text], k=K))[0]
        if len(vanilla_candidates) == 0:
            logger.warning(f"No candidates found for {text}")
            return None

        # if len(vanilla_candidates) == 0:
        #     logger.debug(
        #         f"No candidates found for {text} with similarity >= {self.min_similarity}"
        #     )
        #     # apply rewrites and look for another match
        #     rewritten_text = apply_match_retry_rewrites(text)
        #     if rewritten_text is not None:
        #         return await self._get_best_candidate(rewritten_text, is_composite)
        #     return None

        # apply word overrides (e.g. if term is "modulator", give explicit UMLS match)
        # doing now since it can affect scoring
        candidates = apply_umls_word_overrides(text, vanilla_candidates)

        scored_candidates = [
            (candidate, score_candidate(candidate, is_composite))
            for candidate in candidates
        ]
        # sort by score
        sorted_candidates = sorted(
            scored_candidates, key=lambda sc: sc[1], reverse=True
        )
        logger.debug("Sorted candidates: %s", sorted_candidates)
        return sorted_candidates[0]

    @overrides(AbstractCandidateSelector)
    async def select_candidate(
        self, text: str, is_composite: bool = False
    ) -> EntityWithScore | None:
        """
        Select the best candidate for a mention
        """
        res = await self._get_best_candidate(text, is_composite)

        if res is None:
            return None

        candidate, score = res
        top_canonical = candidate_to_canonical(candidate)
        return top_canonical, score

    async def select_candidate_by_vector(
        self, vector: torch.Tensor, min_similarity: float = 0.85
    ) -> EntityWithScore | None:
        """
        Select the best candidate for a mention
        """
        candidates = await self.candidate_generator.get_candidates(
            vector, K, min_similarity
        )

        if len(candidates) == 0:
            return None

        candidate = candidates[0]

        top_canonical = candidate_to_canonical(candidate)
        return top_canonical, candidate.similarity

    @overrides(AbstractCandidateSelector)
    async def select_candidate_from_entity(
        self, entity: DocEntity, is_composite: bool = False
    ) -> EntityWithScore | None:
        return await self.select_candidate(entity.term, is_composite)

    @overrides(AbstractCandidateSelector)
    async def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidate = await self.select_candidate(entity.term)

        if candidate is None:
            return None

        return candidate[0]
