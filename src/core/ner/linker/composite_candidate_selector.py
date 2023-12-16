from typing import Sequence
from pydash import flatten, uniq
from spacy.tokens import Token
import logging
from scispacy.candidate_generation import MentionCandidate
import torch

from constants.patterns.iupac import is_iupac
from core.ner.types import CanonicalEntity, DocEntity
from data.domain.biomedical.umls import clean_umls_name

from .candidate_selector import CandidateSelector
from .utils import generate_ngram_phrases, get_orthogonal_members, l1_normalize

NGRAMS_N = 2
MIN_WORD_LENGTH = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_COMPOSITE_SIMILARITY = 1.0


class CompositeCandidateSelector(CandidateSelector):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(id, '\\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;

    - Certain gene names are matched naively (e.g. "cell" -> CEL gene, tho that one in particular is suppressed)

    TODO:
        pde-v inhibitor  - works of pde-v but not pde v or pdev
        bace 2 inhibitor - base2
        glp-2 agonist - works with dash
        'at1 receptor antagonist'
        "hyperproliferative disease cancer"
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.select_candidate = super().select_candidate

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

    @classmethod
    def _get_ngrams(
        cls, tokens: Sequence[Token], n: int
    ) -> list[tuple[str, list[float]]]:
        """
        Get all ngrams in a text
        """
        # if fewer words than n, just return words
        # (this is expedient but probably confusing)
        if n == 1 or len(tokens) < n:
            return [(token.text, token.vector.tolist()) for token in tokens]

        ngrams = generate_ngram_phrases(tokens, n)
        return ngrams

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
        self, members: Sequence[tuple[CanonicalEntity, list[float]]]
    ) -> CanonicalEntity | None:
        """
        Form a composite from a list of member entities
        """

        if len(members) == 0:
            return None

        # if just a single composite match, treat it like a non-composite match
        if len(members) == 1:
            return members[0][0]

        ortho_member_idx = get_orthogonal_members(
            torch.stack([torch.tensor(m[1]) for m in members])
        )

        ortho_members = [members[i][0] for i in ortho_member_idx]

        # sorted for the sake of consist composite ids
        ids = sorted([c.id for c in ortho_members])

        # form name from comprising candidates
        name = self._form_composite_name(ortho_members)

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=name,
            # description=..., # TODO: composite description
            # aliases=... # TODO: all permutations
        )

    # def _generate_composite(
    #     self,
    #     tokens: Sequence[Token],
    #     ngram_entity_map: Mapping[str, CanonicalEntity],
    # ) -> CanonicalEntity | None:
    #     """
    #     Generate a composite candidate from a mention text

    #     Args:
    #         mention_text (str): Mention text
    #         ngram_entity_map (dict[str, MentionCandidate]): word-to-candidate map
    #     """

    #     def get_composite_candidates(tokens: Sequence[Token]) -> list[CanonicalEntity]:
    #         """
    #         Recursive function to see if the first ngram has a match, then the first n-1, etc.
    #         """
    #         if len(tokens) == 0:
    #             return []

    #         if len(tokens) >= NGRAMS_N:
    #             ngram = "".join([t.text_with_ws for t in tokens[0:NGRAMS_N]])
    #             if ngram in ngram_entity_map:
    #                 remaining_words = tokens[NGRAMS_N:]
    #                 return [
    #                     ngram_entity_map[ngram],
    #                     *get_composite_candidates(remaining_words),
    #                 ]

    #         # otherwise, let's map only the first word
    #         remaining_words = tokens[1:]
    #         if tokens[0] in ngram_entity_map:
    #             return [
    #                 ngram_entity_map[tokens[0].text],
    #                 *get_composite_candidates(remaining_words),
    #             ]

    #         # otherwise, no match. create a fake MentionCandidate.
    #         return [
    #             # concept_id is the word itself, so
    #             # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
    #             CanonicalEntity(
    #                 name=tokens[0].lower_,
    #                 id=tokens[0].lower_,
    #             ),
    #             *get_composite_candidates(remaining_words),
    #         ]

    #     candidates = get_composite_candidates(tokens)

    #     return self._form_composite(candidates)

    def _get_composite_candidates(
        self, matchless_entity: DocEntity
    ) -> list[MentionCandidate]:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_entity (DocEntity): a doc entity (NER span)
        """
        if matchless_entity.spacy_doc is None:
            raise ValueError("Entity must have a spacy_doc")

        # only non-punct tokens
        def _is_composable(t: Token) -> bool:
            # TODO: get POS tagging working with transformer
            # TODO: non-competitive blah blah antagoinst - keep dashed terms together
            return t.pos_ != "PUNCT" and t.text != "-"

        tokens = [t for t in matchless_entity.spacy_doc if _is_composable(t)]

        # create 1 and 2grams
        matchless_ngrams = uniq(
            flatten([self._get_ngrams(tokens, i + 1) for i in range(NGRAMS_N)])
        )

        # get candidates for all ngrams
        cands = flatten([self._get_candidates(ngram[0]) for ngram in matchless_ngrams])

        return cands

    def get_composite_pieces(
        self,
        vector,
        candidates,
        matches: list[tuple[CanonicalEntity, list[float]]] = [],
        last_score: float | None = None,
    ):
        norm_vector = l1_normalize(torch.tensor(vector))

        match, score, top_vector = self._get_best_canonical(
            norm_vector.tolist(), candidates
        )

        if not match or (len(matches) > 0 and match.id == matches[-1][0].id):
            return matches

        if (
            last_score is not None
            and last_score < MIN_COMPOSITE_SIMILARITY
            and score < MIN_COMPOSITE_SIMILARITY
        ):
            return matches

        residual = norm_vector - torch.tensor(top_vector)
        new_matches = (
            matches + [(match, top_vector)]
            if score > MIN_COMPOSITE_SIMILARITY
            else matches
        )
        return self.get_composite_pieces(
            residual.tolist(), candidates, new_matches, score
        )

    def __call__(self, entity: DocEntity) -> tuple[CanonicalEntity | None, list[float]]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        if not entity.vector or not entity.spacy_doc:
            raise ValueError("Entity must have a vector")

        candidates = self._get_candidates(entity.normalized_term)

        norm_vector = l1_normalize(
            torch.tensor(
                0.9 * torch.tensor(entity.vector)
                + 0.1 * torch.tensor(entity.spacy_doc.vector)
            )
        )
        match, score, vector = self._get_best_canonical(
            norm_vector.tolist(), candidates
        )

        if score >= self.min_similarity:
            return (match, vector)

        composition_candidates = self._get_composite_candidates(entity)

        matches = self.get_composite_pieces(
            norm_vector, composition_candidates + candidates
        )

        # vector of composite
        match_vector = torch.mean(torch.stack([torch.tensor(m[1]) for m in matches]))

        # loss = entity.vector - match_vector

        return (self._form_composite(matches), match_vector.tolist())
