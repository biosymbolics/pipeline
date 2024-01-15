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
        self,
        *args,
        min_similarity: float = 0.85,
        min_composite_similarity: float = 0.7,
        min_word_length: int = 3,  # higher for non-semantic composite
        **kwargs
    ):
        super().__init__(*args, min_similarity=min_similarity, **kwargs)
        self.min_composite_similarity = min_composite_similarity
        self.min_word_length = min_word_length

    def _select_composite_members(
        self, members: Sequence[EntityWithScore]
    ) -> list[EntityWithScore]:
        real_members = [m for m in members if not m[0].is_fake]

        # Partial match if non-matched words, and only a single candidate (TODO: revisit)
        is_partial = (
            # has 1+ fake members (i.e. unmatched)
            len(real_members) < len(members)
            # and only one real candidate match
            and len(real_members) == 1
        )

        # if partial match, include *all* candidates, which includes the faked ones
        # "UNMATCHED inhibitor" will have a name and id that reflects the unmatched word
        if is_partial:
            return list(members)

        # if we have 1+ preferred candidates, return those
        # this prevents composites like C0024579|C0441833 ("Maleimides Groups") - wherein "Group" offers little value
        preferred = [
            m
            for m in real_members
            if has_intersection(m[0].types, list(MOST_PREFERRED_UMLS_TYPES.keys()))
        ]
        if len(preferred) >= 1:
            return preferred

        # else, we're going to drop unmatched words
        # e.g. "cpla (2)-selective inhibitor" -> "cpla inhibitor"

        return real_members

    def _generate_composite(
        self,
        tokens: Sequence[str],
        token_entity_map: Mapping[str, EntityWithScore],
    ) -> EntityWithScore | None:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            token_entity_map (dict[str, CanonicalEntity]): word-to-entity map
        """

        def get_composite_candidate(token: str) -> EntityWithScore:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if token in token_entity_map:
                return token_entity_map[token]

            # otherwise, no match. create a fake CanonicalEntity.
            return (
                # concept_id is the word itself, so composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                CanonicalEntity(id=token.lower(), name=token.lower()),
                # TODO: should be the mean of all candidates, or something?
                self.min_composite_similarity - 0.1,
            )

        if len(tokens) == 0:
            return None

        member_scores = [get_composite_candidate(t) for t in tokens]
        selected = self._select_composite_members(member_scores)

        if len(selected) == 0:
            return None

        composite_members = [m[0] for m in selected]
        composite_canonical = form_composite_entity(composite_members, self.kb)
        composite_score = sum([m[1] for m in selected]) / len(selected)

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
        tokens = entity.normalized_term.split(" ")

        token_entity_map = {
            t: self.select_candidate(t)
            for t in tokens
            # avoid weird matches for single characters/nums & words like "of"
            if len(t) > self.min_word_length
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
        # get initial non-composite match
        res = self.select_candidate(entity.normalized_term)

        if res is None:
            match, match_score = None, 0.0
        else:
            match, match_score = res

        # if score is sufficient, or if it's not a composite candidate, return
        if match_score >= (self.min_similarity + 0.05) or not is_composite_eligible(
            entity
        ):
            return match

        res = self.generate_candidate(entity)
        comp_match, comp_score = res or (None, 0.0)

        if comp_score == 0.0 and match_score == 0.0:
            # no match, composite or direct
            return None

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
