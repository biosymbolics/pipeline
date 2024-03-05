import re
from typing import Mapping, Sequence
from pydash import compact, omit_by
from spacy.tokens import Doc, Span, Token
import logging
import torch
from spacy.lang.en import stop_words

from core.ner.linker.semantic_candidate_selector import SemanticCandidateSelector
from core.ner.linker.types import EntityWithScoreVector
from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from utils.string import (
    generate_ngram_phrases,
    generate_ngram_phrases_from_doc,
    tokens_to_string,
)
from utils.tensor import combine_tensors

from .types import AbstractCompositeCandidateSelector
from .utils import form_composite_entity, is_composite_eligible


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EMPTY = (None, 0.0, None)


class CompositeSemanticCandidateSelector(
    SemanticCandidateSelector, AbstractCompositeCandidateSelector
):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate
    """

    def __init__(
        self, *args, min_composite_similarity: float = 0.9, ngrams_n: int = 3, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_composite_similarity = min_composite_similarity
        self.ngrams_n = ngrams_n

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

            effective_ngrams_n = min(self.ngrams_n, len(tokens) - 1)

            if effective_ngrams_n > 0:
                ngrams = generate_ngram_phrases(
                    tuple([t.text for t in tokens]), effective_ngrams_n
                )
                ngram_matches: list[EntityWithScoreVector] = sorted(
                    compact([ngram_entity_map.get(ng) for ng in ngrams]),
                    key=lambda m: len(m[0].name.split(" ")),
                    reverse=True,
                )
                if len(ngram_matches) > 0 and ngram_matches[0][1] is not None:
                    best_match = ngram_matches[0]
                    remainder_idx = len(best_match[0].name.split(" "))

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
                    self.min_composite_similarity,  # TODO: should be the mean of all candidates, or something?
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
            form_composite_entity(composite_members, self.kb),
            avg_score,
            comp_match_vector,
        )

    @overrides(AbstractCompositeCandidateSelector)
    def generate_candidate(self, entity: DocEntity) -> EntityWithScoreVector | None:
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
            if context_vector is not None:
                vectors = [
                    combine_tensors(torch.tensor(d.vector), context_vector, 0.95)
                    for d in ngram_docs
                ]
            else:
                vectors = [torch.tensor(d.vector) for d in ngram_docs]

            return ngram_docs, vectors

        non_stopwords = " ".join(
            [
                w
                for w in entity.normalized_term.split(" ")
                if w not in stop_words.STOP_WORDS
            ]
        )
        doc = self.nlp(non_stopwords)

        ngrams, ngram_vectors = generate_ngram_spans(doc, torch.tensor(entity.vector))
        ngram_entity_map = {
            t.text: self.select_candidate(t.text, vector, is_composite=True)
            for t, vector in zip(ngrams, ngram_vectors)
            if len(t.text) > 1  # avoid weird matches for single characters/nums
        }
        print(
            {
                k: (v[0].name, v[1])
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

    @overrides(SemanticCandidateSelector)
    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        # get initial non-composite match
        match, match_score, _ = (
            super().select_candidate_from_entity(entity, is_composite=False) or EMPTY
        )

        # if high enough score, or not a composite candidate, return
        if match_score >= self.min_similarity or not is_composite_eligible(entity):
            return match

        # else, generate a composite candidate
        comp_match, comp_score, _ = self.generate_candidate(entity) or EMPTY

        # if composite and direct matches bad, no match.
        if False and (
            comp_score < self.min_similarity and match_score < self.min_similarity
        ):
            logger.warning("No match for %s", entity.normalized_term)
            return None

        if comp_score > match_score:
            logger.info(
                "Composite has higher score (%s vs %s)", comp_score, match_score
            )
            return comp_match

        logger.info(
            "Non-composite has higher score (%s vs %s)", match_score, comp_score
        )
        return match
