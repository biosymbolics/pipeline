from typing import Mapping, Sequence
from pydash import omit_by
import logging
from spacy.tokens import Doc, Span, Token

import torch

from core.ner.linker.types import EntityWithScore, EntityWithScoreVector
from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from utils.string import generate_ngram_phrases_from_doc, tokens_to_string
from utils.tensor import combine_tensors

from .candidate_selector import CandidateSelector
from .composite_utils import (
    form_composite_entity,
    is_composite_eligible,
    select_composite_members,
)
from .types import AbstractCompositeCandidateSelector


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CompositeCandidateSelector(CandidateSelector, AbstractCompositeCandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    """

    def __init__(
        self,
        *args,
        min_similarity: float = 0.85,
        min_composite_similarity: float = 0.8,
        min_word_length: int = 3,  # higher for non-semantic composite
        ngrams_n: int = 3,
        **kwargs
    ):
        super().__init__(*args, min_similarity=min_similarity, **kwargs)
        self.min_composite_similarity = min_composite_similarity
        self.min_word_length = min_word_length
        self.ngrams_n = ngrams_n

    def _generate_composite(
        self,
        words: Sequence[str],
        token_entity_map: Mapping[str, EntityWithScore],
    ) -> EntityWithScore | None:
        """
        Generate a composite candidate from a mention text

        Args:
            words (str): Mention words
            token_entity_map (dict[str, CanonicalEntity]): word-to-entity map
        """

        def get_composite_candidate(token: str) -> EntityWithScore:
            if token in token_entity_map:
                return token_entity_map[token]

            # create a fake CanonicalEntity.
            return (
                # concept_id is the word itself
                # so composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                CanonicalEntity(id=token.lower(), name=token.lower()),
                self.min_composite_similarity - 0.1,
            )

        if len(words) == 0:
            return None

        members_with_scores = [get_composite_candidate(w) for w in words]
        score_map = {m[0].name: m[1] for m in members_with_scores}
        composite_members = select_composite_members(
            [m[0] for m in members_with_scores]
        )

        if len(composite_members) == 0:
            return None

        composite_canonical = form_composite_entity(composite_members)
        composite_score = sum([score_map[m.name] for m in composite_members]) / len(
            composite_members
        )

        return (composite_canonical, composite_score)

    async def _optimize_composite(
        self, composite: EntityWithScore, original_name: str
    ) -> EntityWithScore | None:
        """
        Taking the new composite names, see if there is now a singular match
        (e.g. a composite name might be "SGLT2 inhibitor", comprised of two candidates, for which a single match exists)
        """
        if composite[0].name.lower() == original_name.lower():
            # if the name hasn't changed, pointless to attempt re-match
            return composite

        direct_match = await self.select_candidate(composite[0].name)

        if direct_match is None:
            return composite

        return direct_match

    async def do_thing(self, entity: DocEntity) -> EntityWithScoreVector | None:
        """
        Generate a composite candidate from a doc entity
        """

        def generate_ngram_spans(
            doc: Doc, context_vector: torch.Tensor | None
        ) -> tuple[list[Span], list[torch.Tensor]]:
            """
            Get tokens and vectors from a doc entity
            """
            ngram_docs = generate_ngram_phrases_from_doc(doc, self.ngrams_n)

            # if the entity has a vector, combine with newly created token vectors
            # to add context for semantic similarity comparison
            # TODO: subtract this before calculating residual?
            if context_vector is not None:
                vectors = [
                    combine_tensors(torch.tensor(d.vector), context_vector, 0.8)
                    for d in ngram_docs
                ]
            else:
                vectors = [torch.tensor(d.vector) for d in ngram_docs]

            return ngram_docs, vectors

        if entity.spacy_doc is None:
            raise ValueError("Entity must have a spacy doc")
        else:
            doc = entity.spacy_doc

        ngrams, ngram_vectors = generate_ngram_spans(doc, torch.tensor(entity.vector))
        ngram_entity_map = {
            t.text: await self.select_candidate(t.text, is_composite=True)
            for t, vector in zip(ngrams, ngram_vectors)
            if len(t.text) > 1  # avoid weird matches for single characters/nums
        }
        logger.info(
            "Ngram entity map: %s",
            {
                k: (v[0].name, v[1], v[1] >= self.min_composite_similarity)
                for k, v in ngram_entity_map.items()
                if v is not None
            },
        )
        return self._generate_composite_from_ngrams(
            list(doc),
            omit_by(
                ngram_entity_map,
                lambda v: v is None or v[1] < self.min_composite_similarity,
            ),
        )

    @overrides(AbstractCompositeCandidateSelector)
    async def generate_candidate(self, entity: DocEntity) -> EntityWithScore | None:
        """
        Select compsosite candidate for a mention text (i.e. analog to select_candidate_from_entity)
        """
        # len check - avoid weird matches for single characters/nums & words like "of"
        tokens = [
            t
            for t in entity.normalized_term.split(" ")
            if len(t) > self.min_word_length
        ]

        token_entity_map = {
            t: self.select_candidate(t, is_composite=True) for t in tokens
        }
        composite = self._generate_composite(
            tokens,
            omit_by(
                token_entity_map,
                lambda v: v is None or v[1] < self.min_composite_similarity,
            ),
        )

        if composite is None:
            return None

        return await self._optimize_composite(composite, entity.normalized_term)

    def _generate_composite_from_ngrams(
        self,
        tokens: Sequence[Token | Span],
        ngram_entity_map: Mapping[str, EntityWithScoreVector],
    ) -> EntityWithScoreVector:
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
        comp_match_vector = torch.mean(torch.stack([m[2] for m in composites]), dim=0)
        composite_members = [c[0] for c in composites]

        return (
            form_composite_entity(composite_members),
            avg_score,
            comp_match_vector,
        )

    async def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        # attempt direct/non-composite match
        res = await self.select_candidate_from_entity(entity, is_composite=False)

        if res is None:
            match, match_score = None, 0.0
        else:
            match, match_score = res

        # if score is sufficient, or if it's not a composite candidate, return
        if match_score >= (self.min_similarity + 0.05) or not is_composite_eligible(
            entity
        ):
            return match

        # generate composite candidate
        res = await self.generate_candidate(entity)
        comp_match, comp_score = res or (None, 0.0)

        # if composite and direct matches are bad, no match.
        if comp_score < self.min_similarity and match_score < self.min_similarity:
            return None

        if comp_score > match_score:
            logger.debug(
                "Composite has higher score (%s vs %s)", comp_score, match_score
            )
            return comp_match

        logger.debug(
            "Non-composite has higher score (%s vs %s)", match_score, comp_score
        )
        return match
