from typing import Mapping, Sequence
from pydash import omit_by
import logging

from core.ner.linker.candidate_selector import CandidateSelector
from core.ner.linker.types import EntityScore
from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides

from .types import AbstractCompositeCandidateSelector
from .utils import form_composite_entity, is_composite_eligible


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CompositeCandidateSelector(CandidateSelector, AbstractCompositeCandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)

    TODO:
        pde-v inhibitor  - works of pde-v but not pde v or pdev
        bace 2 inhibitor - base2
        glp-2 agonist - works with dash
        'at1 receptor antagonist'
        "hyperproliferative disease cancer"
    """

    def __init__(
        self, *args, ngrams_n: int = 1, min_composite_similarity: float = 0.85, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_composite_similarity = min_composite_similarity
        self.ngrams_n = ngrams_n

    def _generate_composite(
        self,
        tokens: Sequence[str],
        ngram_entity_map: Mapping[str, EntityScore],
    ) -> EntityScore | None:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            ngram_entity_map (dict[str, CanonicalEntity]): word-to-entity map

        TODO: remove ngram if ultimately only using N of 1
        """

        def get_composite_candidates(tokens: Sequence[str]) -> list[EntityScore]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(tokens) == 0:
                return []

            if len(tokens) >= self.ngrams_n:
                ngram = "".join([t for t in tokens[0 : self.ngrams_n]])
                if ngram in ngram_entity_map:
                    remaining_words = tokens[self.ngrams_n :]
                    return [
                        ngram_entity_map[ngram],
                        *get_composite_candidates(remaining_words),
                    ]

            # otherwise, let's map only the first word
            remaining_words = tokens[1:]
            if tokens[0] in ngram_entity_map:
                return [
                    ngram_entity_map[tokens[0]],
                    *get_composite_candidates(remaining_words),
                ]

            # otherwise, no match. create a fake CanonicalEntity.
            return [
                # concept_id is the word itself, so
                # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                (
                    CanonicalEntity(
                        id=tokens[0].lower(),
                        name=tokens[0].lower(),
                    ),
                    self.min_composite_similarity,  # TODO: should be the mean of all candidates, or something?
                ),
                *get_composite_candidates(remaining_words),
            ]

        composites = get_composite_candidates(tokens)

        if len(composites) == 0:
            return None

        composite_members = [c[0] for c in composites]
        composite_canonical = form_composite_entity(composite_members, self.kb)
        composite_score = sum([m[1] for m in composites]) / len(composites)

        return (composite_canonical, composite_score)

    @overrides(AbstractCompositeCandidateSelector)
    def generate_candidate(self, entity: DocEntity) -> EntityScore | None:
        """
        Select compsosite candidate for a mention text (i.e. analog to select_candidate_from_entity)
        """
        tokens = entity.normalized_term.split(" ")

        ngram_entity_map = {
            t: super().select_candidate(t)
            for t in tokens
            if len(t) > 1  # avoid weird matches for single characters/nums
        }
        return self._generate_composite(
            tokens,
            omit_by(
                ngram_entity_map,
                lambda v: v is None or v[1] < self.min_composite_similarity,
            ),
        )

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        # get initial non-composite match
        res = super().select_candidate(entity.normalized_term)

        if res is None:
            match, match_score = None, 0.0
        else:
            match, match_score = res

        # if score is sufficient, or if it's not a composite candidate, return
        if match_score >= self.min_similarity or not is_composite_eligible(entity):
            return match

        res = self.generate_candidate(entity)
        comp_match, comp_score = res or (None, 0.0)

        if comp_score > match_score:
            logger.info(
                "Returning composite match with higher score (%s vs %s)",
                comp_score,
                match_score,
            )
            return comp_match

        logger.info(
            "Returning non-composite match with higher score (%s vs %s)",
            match_score,
            comp_score,
        )
        return match
