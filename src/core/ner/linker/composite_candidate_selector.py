from typing import Mapping, Sequence
from pydash import omit_by
from spacy.tokens import Span, Token
import logging
import torch

from constants.patterns.iupac import is_iupac
from core.ner.types import CanonicalEntity, DocEntity
from data.domain.biomedical.umls import clean_umls_name

from .semantic_candidate_selector import SemanticCandidateSelector
from .utils import join_punctuated_tokens

NGRAMS_N = 1
MIN_WORD_LENGTH = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_COMPOSITE_SIMILARITY = 1.0


SelectedEntityScoreVector = tuple[CanonicalEntity | None, float, torch.Tensor]
EntityScoreVector = tuple[CanonicalEntity, float, torch.Tensor]


class CompositeCandidateSelector(SemanticCandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(id, '\\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;


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
        min_composite_similarity: float = MIN_COMPOSITE_SIMILARITY,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_composite_similarity = min_composite_similarity

    @staticmethod
    def _is_composite_eligible(entity: DocEntity) -> bool:
        """
        Is a text a composite candidate?

        - False if it's an IUPAC name, which are just too easy to mangle (e.g. 3'-azido-2',3'-dideoxyuridine matching 'C little e')
        - false if it's too short (a single token or word)
        - Otherwise true
        """
        tokens = entity.spacy_doc or entity.normalized_term.split(" ")
        if is_iupac(entity.normalized_term):
            return False
        if len(tokens) <= MIN_WORD_LENGTH:
            return False
        return True

    def _form_composite_name(self, member_candidates: Sequence[CanonicalEntity]) -> str:
        """
        Form a composite name from the candidates from which it is comprised
        """

        def get_name_part(c: CanonicalEntity):
            if c.id in self.kb.cui_to_entity:
                ce = self.kb.cui_to_entity[c.id]
                return clean_umls_name(
                    ce.concept_id, ce.canonical_name, ce.aliases, ce.types, True
                )
            return c.name

        name = " ".join([get_name_part(c) for c in member_candidates])
        return name

    def _form_composite(
        self, members: Sequence[EntityScoreVector]
    ) -> CanonicalEntity | None:
        """
        Form a composite from a list of member entities
        """

        if len(members) == 0:
            return None

        # if just a single composite match, treat it like a non-composite match
        if len(members) == 1:
            return members[0][0]

        # hard to get this right... distances are unexpected.
        # ortho_member_idx = get_orthogonal_members(torch.stack([m[2] for m in members]))
        # ortho_members = [members[i][0] for i in ortho_member_idx]

        # sorted for the sake of consist composite ids
        ids = sorted([m[0].id for m in members if m[0].id is not None])

        # form name from comprising candidates
        name = self._form_composite_name([m[0] for m in members])

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=name,
            # description=..., # TODO: composite description
            # aliases=... # TODO: all permutations
        )

    def _generate_composite(
        self,
        tokens: Sequence[Token | Span],
        ngram_entity_map: Mapping[str, tuple[CanonicalEntity, float, torch.Tensor]],
    ) -> SelectedEntityScoreVector:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            ngram_entity_map (dict[str, MentionCandidate]): word-to-candidate map

        TODO: remove ngram if ultimately only using N of 1
        """

        def get_composite_candidates(
            tokens: Sequence[Token | Span],
        ) -> list[tuple[CanonicalEntity, float, torch.Tensor]]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(tokens) == 0:
                return []

            if len(tokens) >= NGRAMS_N:
                ngram = "".join([t.text_with_ws for t in tokens[0:NGRAMS_N]])
                if ngram in ngram_entity_map:
                    remaining_words = tokens[NGRAMS_N:]
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

            # otherwise, no match. create a fake MentionCandidate.
            return [
                # concept_id is the word itself, so
                # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                (
                    CanonicalEntity(
                        id=tokens[0].text.lower(),
                        name=tokens[0].text.lower(),
                    ),
                    MIN_COMPOSITE_SIMILARITY,  # TODO: should be the mean of all candidates, or something?
                    torch.tensor(tokens[0].vector),
                ),
                *get_composite_candidates(remaining_words),
            ]

        composites = get_composite_candidates(tokens)
        avg_score = sum([m[1] for m in composites]) / len(composites)
        comp_match_vector = torch.mean(torch.stack([m[2] for m in composites]))

        return (self._form_composite(composites), avg_score, comp_match_vector)

    def select_composite_candidate(
        self, entity: DocEntity
    ) -> SelectedEntityScoreVector:
        """
        Select compsosite candidate for a mention text (i.e. analog to select_candidate_from_entity)
        """
        if not entity.spacy_doc:
            raise ValueError("Entity must have a vector")

        # join tokens presumed to be joined by punctuation, e.g. ['non', '-', 'competitive'] -> "non-competitive"
        tokens = join_punctuated_tokens(entity.spacy_doc)

        ngram_entity_map = {
            t.text: self.select_candidate(
                t.text, torch.tensor(t.vector), torch.tensor(entity.spacy_doc.vector)
            )
            for t in tokens
            if len(t) > 1  # avoid weird matches for single characters/nums
        }
        return self._generate_composite(
            tokens,
            omit_by(
                ngram_entity_map,
                lambda v: v[0] is None or v[1] < self.min_composite_similarity,
            ),
        )

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        if not entity.spacy_doc:
            raise ValueError("Entity must have a vector")

        # get initial non-composite match
        match, match_score, vector = super().select_candidate_from_entity(entity)

        # if score is sufficient, or if it's not a composite candidate, return
        if match_score >= self.min_similarity or not self._is_composite_eligible(
            entity
        ):
            return match

        # join tokens presumed to be joined by punctuation, e.g. ['non', '-', 'competitive'] -> "non-competitive"
        tokens = join_punctuated_tokens(entity.spacy_doc)

        ngram_entity_map = {
            t.text: self.select_candidate(
                t.text, torch.tensor(t.vector), torch.tensor(entity.spacy_doc.vector)
            )
            for t in tokens
            if len(t) > 1  # avoid weird matches for single characters/nums
        }
        comp_match, comp_score, comp_vector = self._generate_composite(
            tokens,
            omit_by(
                ngram_entity_map,
                lambda v: v[0] is None or v[1] < self.min_composite_similarity,
            ),
        )

        # loss = cosine_dist(entity.vector, match_vector)

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
