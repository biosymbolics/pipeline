from typing import Mapping, Sequence
from pydash import omit_by
import logging

from constants.umls import MOST_PREFERRED_UMLS_TYPES
from core.ner.linker.candidate_selector import CandidateSelector
from core.ner.linker.types import EntityWithScore
from core.ner.types import CanonicalEntity, DocEntity
from utils.classes import overrides
from utils.list import has_intersection

from .types import AbstractCompositeCandidateSelector
from .utils import (
    form_composite_entity,
    is_composite_eligible,
    select_composite_members,
)


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
        self,
        *args,
        min_similarity: float = 0.85,
        min_composite_similarity: float = 0.8,
        min_word_length: int = 3,  # higher for non-semantic composite
        **kwargs
    ):
        super().__init__(*args, min_similarity=min_similarity, **kwargs)
        self.min_composite_similarity = min_composite_similarity
        self.min_word_length = min_word_length

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

        composite_canonical = form_composite_entity(composite_members, self.kb)
        composite_score = sum([score_map[m.name] for m in composite_members]) / len(
            composite_members
        )

        return (composite_canonical, composite_score)

    def _optimize_composite(
        self, composite: EntityWithScore, original_name: str
    ) -> EntityWithScore | None:
        """
        Taking the new composite names, see if there is now a singular match
        (e.g. a composite name might be "SGLT2 inhibitor", comprised of two candidates, for which a single match exists)
        """
        if composite[0].name.lower() == original_name.lower():
            # if the name hasn't changed, pointless to attempt re-match
            return composite

        direct_match = self.select_candidate(composite[0].name)

        if direct_match is None:
            return composite

        return direct_match

    @overrides(AbstractCompositeCandidateSelector)
    def generate_candidate(self, entity: DocEntity) -> EntityWithScore | None:
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

        return self._optimize_composite(composite, entity.normalized_term)

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        # attempt direct/non-composite match
        res = self.select_candidate_from_entity(entity, is_composite=False)

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
        res = self.generate_candidate(entity)
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
