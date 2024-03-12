import logging
from typing import AsyncIterable, AsyncIterator, Iterable, Iterator
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
K = 10


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateSelector(AbstractCandidateSelector):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    Returns the candidate with the highest similarity score

    TODO: multiple locations of min_similarity!
    """

    @overrides(AbstractCandidateSelector)
    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        min_similarity: float = MIN_SIMILARITY,
    ):
        """
        Initialize candidate selector

        Use `create` to instantiate with async dependencies
        """
        self.min_similarity = min_similarity
        self.candidate_generator = candidate_generator

    @classmethod
    @overrides(AbstractCandidateSelector)
    async def create(cls, *args, **kwargs):
        candidate_generator = await CandidateGenerator.create()
        return cls(candidate_generator=candidate_generator, *args, **kwargs)

    def _score_candidates(
        self,
        candidates: list[MentionCandidate],
        mention_vector: torch.Tensor,
        is_composite: bool,
    ) -> list[tuple[MentionCandidate, float]]:
        """
        Score & sort candidates
        """
        scored_candidates = [
            (
                candidate,
                score_candidate(candidate, mention_vector, is_composite),
            )
            for candidate in candidates
        ]
        # sort by score
        sorted_candidates = sorted(
            scored_candidates, key=lambda sc: sc[1], reverse=True
        )
        return sorted_candidates

    @overrides(AbstractCandidateSelector)
    async def select_candidate(
        self,
        mention: str,
        vector: torch.Tensor | None = None,
        min_similarity: float = 0.85,
        is_composite: bool = False,
    ) -> EntityWithScore | None:
        """
        Select the best candidate for a mention
        """
        candidates, mention_vec = await self.candidate_generator.get_candidates(
            mention, vector, K, min_similarity
        )

        if len(candidates) == 0:
            logger.debug(
                f"No candidates found for {mention} with similarity >= {self.min_similarity}"
            )
            # apply rewrites and look for another match
            rewritten_text = apply_match_retry_rewrites(mention)
            if rewritten_text is not None and rewritten_text != mention:
                return await self.select_candidate(
                    rewritten_text, None, min_similarity, is_composite
                )
            return None

        candidates = apply_umls_word_overrides(mention, candidates)

        scored_candidates = self._score_candidates(
            candidates, mention_vec, is_composite
        )
        candidate, score = scored_candidates[0]

        return candidate_to_canonical(candidate), score

    async def select_candidates(
        self,
        mentions: list[str],
        vectors: list[torch.Tensor] | None = None,
        min_similarity: float = 0.85,
        is_composite: bool = False,
    ) -> AsyncIterator[EntityWithScore | None]:
        """
        Select the best candidate for a mention
        """
        candidate_sets = [
            ci
            async for ci in self.candidate_generator(
                mentions, vectors, K, min_similarity
            )
        ]
        for (candidates, mention_vec), mention in zip(candidate_sets, mentions):
            if len(candidates) > 0:
                candidates = apply_umls_word_overrides(mention, candidates)
                scored_candidates = self._score_candidates(
                    candidates, mention_vec, is_composite
                )
                candidate, score = scored_candidates[0]

                yield candidate_to_canonical(candidate), score
            else:
                yield None

    @overrides(AbstractCandidateSelector)
    def select_candidates_from_entities(
        self,
        entities: Iterable[DocEntity],
        min_similarity: float = 0.85,
        is_composite: bool = False,
    ) -> AsyncIterable[EntityWithScore | None]:
        """
        Generate & select candidates for a list of mention texts
        """
        ents = list(entities)
        mentions = [ent.normalized_term for ent in ents]
        vectors = [torch.tensor(ent.vector) for ent in ents]

        return self.select_candidates(mentions, vectors, min_similarity, is_composite)

    @overrides(AbstractCandidateSelector)
    async def __call__(
        self, entities: Iterable[DocEntity]
    ) -> AsyncIterable[CanonicalEntity | None]:
        async for c in self.select_candidates_from_entities(entities):
            yield c[0] if c else None
