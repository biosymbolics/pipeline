from typing import AsyncIterable, Iterable, Mapping, Sequence
from pydash import compact, omit_by
import logging
from spacy.tokens import Doc, Span, Token
import torch

from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from utils.string import generate_ngram_phrases_from_doc, tokens_to_string
from utils.tensor import combine_tensors, truncated_svd

from .candidate_generator import CandidateGenerator
from .candidate_selector import MIN_SIMILARITY, CandidateSelector
from .composite_utils import (
    form_composite_entity,
    is_composite_eligible,
    select_composite_members,
)
from .types import (
    EntityWithScore,
    EntityWithScoreVector,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CompositeCandidateSelector(CandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    """

    def __init__(
        self,
        candidate_generator: CandidateGenerator,
        *args,
        min_similarity: float = MIN_SIMILARITY,
        min_composite_similarity: float = 0.7,
        min_word_length: int = 2,
        ngrams_n: int = 3,
        **kwargs
    ):
        """
        Initialize composite candidate selector
        Use `create` to instantiate with async dependencies
        """
        super().__init__(
            candidate_generator, *args, min_similarity=min_similarity, **kwargs
        )
        self.min_composite_similarity = min_composite_similarity
        self.min_word_length = min_word_length
        self.ngrams_n = ngrams_n

    @classmethod
    @overrides(CandidateSelector)
    async def create(cls, *args, **kwargs):
        candidate_generator = await CandidateGenerator.create()
        return cls(candidate_generator=candidate_generator, *args, **kwargs)

    async def _generate_composite(self, entity: DocEntity) -> EntityWithScore | None:
        """
        Generate a composite candidate from tokens & ngram map

        Args:
            tokens (Sequence[Token | Span]): tokens to generate composite from
            ngram_entity_map (dict[str, EntityWithScoreVector]): word-to-candidate map
        """

        components = truncated_svd(torch.tensor(entity.vector))
        composites: list[EntityWithScore] = compact(
            [
                c
                async for c in self.select_candidates(
                    None, components, is_composite=True
                )
            ]
        )

        if not composites:
            return None

        avg_score = sum([m[1] for m in composites]) / len(composites)
        composite_members = select_composite_members([c[0] for c in composites])

        return (
            form_composite_entity(composite_members),
            avg_score,
        )

    @overrides(CandidateSelector)
    async def select_candidates_from_entities(
        self,
        entities: Iterable[DocEntity],
    ) -> AsyncIterable[EntityWithScore | None]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        entity_iter = iter(entities)
        # attempt direct/non-composite match
        async for match in super().select_candidates_from_entities(
            entities, is_composite=True
        ):
            entity = next(entity_iter)
            match_score = match[1] if match is not None else 0

            # if score is sufficient, or if it's not a composite candidate, return
            is_eligibile = is_composite_eligible(entity.normalized_term)
            if match_score >= self.min_similarity or not is_eligibile:
                logger.info(
                    "Choose non-composite without check, %s (%s)",
                    entity.normalized_term,
                    match_score,
                )
                yield match
                continue

            # generate composite candidate
            composite = await self._generate_composite(entity)
            composite_score = composite[1] if composite is not None else 0

            if composite_score > match_score:
                logger.info("Chose composite (%s vs %s)", composite_score, match_score)
                yield composite
            else:
                logger.info("Chose non-comp (%s vs %s)", match_score, composite_score)
                yield match

    @overrides(CandidateSelector)
    async def __call__(
        self, entities: Iterable[DocEntity]
    ) -> AsyncIterable[CanonicalEntity | None]:
        """
        Generate & select candidates for a list of mention texts
        """
        async for e in self.select_candidates_from_entities(entities):
            if e is not None:
                yield e[0]
            else:
                yield None
