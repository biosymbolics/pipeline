from typing import Mapping, Sequence
from pydash import omit_by
from spacy.tokens import Span, Token
import logging
import torch

from core.ner.linker.semantic_candidate_selector import SemanticCandidateSelector
from core.ner.linker.types import EntityWithScoreVector
from core.ner.linker.utils import combine_vectors, join_punctuated_tokens
from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides

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
        self, *args, min_composite_similarity: float = 1.0, ngrams_n: int = 1, **kwargs
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
            mention_text (str): Mention text
            ngram_entity_map (dict[str, EntityWithScoreVector]): word-to-candidate map

        TODO: remove ngram if ultimately only using N of 1
        """

        def get_composite_candidates(
            tokens: Sequence[Token | Span],
        ) -> list[EntityWithScoreVector]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(tokens) == 0:
                return []

            if len(tokens) >= self.ngrams_n:
                ngram = "".join([t.text_with_ws for t in tokens[0 : self.ngrams_n]])
                if ngram in ngram_entity_map:
                    remaining_words = tokens[self.ngrams_n :]
                    return [
                        ngram_entity_map[ngram],
                        *get_composite_candidates(remaining_words),
                    ]

            # otherwise, let's map only the first word
            remaining_words = tokens[1:]
            if tokens[0].text in ngram_entity_map:
                return [
                    ngram_entity_map[tokens[0].text],
                    *get_composite_candidates(remaining_words),
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
                *get_composite_candidates(remaining_words),
            ]

        composites = get_composite_candidates(tokens)
        avg_score = sum([m[1] for m in composites]) / len(composites)
        comp_match_vector = torch.mean(torch.stack([m[2] for m in composites]))
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

        if not entity.spacy_doc:
            doc = self.nlp(entity.normalized_term)
            tokens = join_punctuated_tokens(doc)
            vectors = [
                combine_vectors(
                    torch.tensor(t.vector), torch.tensor(entity.vector), 0.9
                )
                for t in tokens
            ]
        else:
            # join tokens presumed to be joined by punctuation, e.g. ['non', '-', 'competitive'] -> "non-competitive"
            tokens = join_punctuated_tokens(entity.spacy_doc)
            vectors = [torch.tensor(t.vector) for t in tokens]

        ngram_entity_map = {
            t.text: self.select_candidate(t.text, vector)
            for t, vector in zip(tokens, vectors)
            if len(t) > 1  # avoid weird matches for single characters/nums
        }
        return self._generate_composite_from_ngrams(
            tokens,
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
        match, match_score, _ = super().select_candidate_from_entity(entity) or EMPTY

        # if high enough score, or not a composite candidate, return
        if match_score >= self.min_similarity or not is_composite_eligible(entity):
            return match

        # else, generate a composite candidate
        comp_match, comp_score, _ = self.generate_candidate(entity) or EMPTY

        if comp_score > match_score:
            logger.debug(
                "Composite has higher score (%s vs %s)", comp_score, match_score
            )
            return comp_match

        logger.debug(
            "Non-composite has higher score (%s vs %s)", match_score, comp_score
        )
        return match
