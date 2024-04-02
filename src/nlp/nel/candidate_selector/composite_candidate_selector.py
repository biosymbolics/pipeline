from typing import AsyncIterable, Iterable, Mapping, Sequence
from pydash import compact, omit_by
import logging
from spacy.tokens import Doc, Span, Token
import torch

from nlp.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from utils.string import generate_ngram_phrases_from_doc, tokens_to_string
from utils.tensor import calc_dynamic_cutoff, combine_tensors, truncated_svd

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

    def _generate_composite_from_ngrams(
        self,
        tokens: Sequence[Token | Span],
        ngram_entity_map: Mapping[str, EntityWithScoreVector],
    ) -> EntityWithScore:
        """
        Generate a composite candidate from tokens & ngram map
        Args:
            tokens (Sequence[Token | Span]): tokens to generate composite from
            ngram_entity_map (dict[str, EntityWithScoreVector]): word-to-candidate map
        """

        def get_composite_candidates(
            tokens: Sequence[Token | Span],
        ) -> list[EntityWithScoreVector]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(tokens) == 0:
                return []

            actual_ngrams_n = min(self.ngrams_n, len(tokens))

            possible_ngrams = [
                (n, tokens_to_string(tokens[0:n]))
                for n in range(actual_ngrams_n, 0, -1)
            ]
            ngram_matches = sorted(
                [
                    (n, ngram_entity_map[ng])
                    for n, ng in possible_ngrams
                    if ng in ngram_entity_map
                ],
                key=lambda m: m[1][1],
                reverse=True,
            )
            if len(ngram_matches) > 0:
                best_match = ngram_matches[0][1]
                remainder_idx = ngram_matches[0][0]

                return [
                    best_match,
                    *get_composite_candidates(tokens[remainder_idx:]),
                ]

            # otherwise, no match. create a fake CanonicalEntity.
            return [
                # concept_id is the word itself, so
                # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                (
                    CanonicalEntity(
                        id=tokens[0].text.lower(),
                        name=tokens[0].text.lower(),
                    ),
                    self.min_composite_similarity,  # TODO: should be the mean of all candidates?
                    torch.tensor(tokens[0].vector),
                ),
                *get_composite_candidates(tokens[1:]),
            ]

        composites = get_composite_candidates(tokens)

        if len(composites) == 0:
            raise ValueError("No composites found")

        avg_score = sum([m[1] for m in composites]) / len(composites)
        composite_members = select_composite_members([c[0] for c in composites])
        return (
            form_composite_entity(composite_members),
            avg_score,
        )

    async def _generate_composite(self, entity: DocEntity) -> EntityWithScore | None:
        """
        Generate a composite candidate from a doc entity
        """

        def generate_ngram_spans(
            doc: Doc, context_vector: torch.Tensor | None
        ) -> tuple[list[str], list[torch.Tensor]]:
            """
            Get tokens and vectors from a doc entity
            """
            ngram_docs = generate_ngram_phrases_from_doc(doc, self.ngrams_n, 2)
            ngram_strings = [ng.text for ng in ngram_docs]

            # if the entity has a vector, combine with newly created token vectors
            # to add context for semantic similarity comparison
            # TODO: subtract this before calculating residual?
            if context_vector is not None:
                vectors = [
                    combine_tensors(torch.tensor(d.vector), context_vector, 0.95)
                    for d in ngram_docs
                ]
            else:
                vectors = [torch.tensor(d.vector) for d in ngram_docs]

            return ngram_strings, vectors

        if entity.spacy_doc is None:
            raise ValueError("Entity must have a spacy doc")

        doc = entity.spacy_doc

        ngrams, ngram_vectors = generate_ngram_spans(doc, torch.tensor(entity.vector))
        ngram_candidates = [
            c
            async for c in self.select_candidates(
                ngrams,
                ngram_vectors,
                self.min_composite_similarity,
                is_composite=True,
            )
        ]
        ngram_entity_map = {
            ng: candidate for ng, candidate in zip(ngrams, ngram_candidates)
        }
        logger.info(
            "Ngram entity map: %s",
            {
                k: (v[0].name, v[1])
                for k, v in ngram_entity_map.items()
                if v is not None
            },
        )
        min_score = calc_dynamic_cutoff(
            [v[1] for v in ngram_entity_map.values() if v is not None], 0
        )
        return self._generate_composite_from_ngrams(
            list(doc),
            omit_by(
                ngram_entity_map,
                lambda v: v is None or v[1] < min_score,
            ),
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
