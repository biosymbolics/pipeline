from typing import Mapping, Sequence
from spacy.lang.en.stop_words import STOP_WORDS
from scispacy.candidate_generation import MentionCandidate
from constants.patterns.iupac import is_iupac

from core.ner.types import CanonicalEntity, DocEntity
from constants.umls import PREFERRED_UMLS_TYPES
from data.domain.biomedical.umls import clean_umls_name
from utils.list import has_intersection
from utils.string import generate_ngram_phrases

from .candidate_selector import CandidateSelector

NGRAMS_N = 2
MIN_WORD_LENGTH = 1


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

    @staticmethod
    def _is_composite_eligible(text: str, canonical: CanonicalEntity | None) -> bool:
        """
        Is a text a composite candidate?

        - False if we have a canonical candidate
        - False if it's an IUPAC name, which are just too easy to mangle (e.g. 3'-azido-2',3'-dideoxyuridine matching 'C little e')
        - Otherwise true
        """
        if canonical is None:
            return False
        if is_iupac(text):
            return False
        return True

    @classmethod
    def _get_words(cls, text: str) -> tuple[str, ...]:
        """
        Get all words in a text, above min length and non-stop-word
        """
        return tuple(
            [
                word
                for word in text.split()
                if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS
            ]
        )

    @classmethod
    def _get_ngrams(cls, text: str, n: int) -> list[str]:
        """
        Get all ngrams in a text
        """
        words = cls._get_words(text)

        # if fewer words than n, just return words
        # (this is expedient but probably confusing)
        if n == 1 or len(words) < n:
            return list(words)

        ngrams = generate_ngram_phrases(words, n)
        return ngrams

    def _form_composite_name(
        self, member_candidates: Sequence[MentionCandidate]
    ) -> str:
        """
        Form a composite name from the candidates from which it is comprised
        """

        def get_name_part(c):
            if c.concept_id in self.kb.cui_to_entity:
                ce = self.kb.cui_to_entity[c.concept_id]
                return clean_umls_name(
                    ce.concept_id, ce.canonical_name, ce.aliases, ce.types, True
                )
            return c.aliases[0]

        name = " ".join([get_name_part(c) for c in member_candidates])
        return name

    def _form_composite(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Form a composite from a list of candidates
        """

        def get_preferred_typed_candidates(
            candidates: Sequence[MentionCandidate],
        ) -> list[MentionCandidate]:
            """
            Get candidates that are of a preferred type
            """
            entities = [self.kb.cui_to_entity.get(c.concept_id) for c in candidates]
            candidate_to_types = {
                c.concept_id: e.types
                for c, e in zip(candidates, entities)
                if e is not None
            }
            preferred_type_candidates = [
                c
                for c in candidates
                if has_intersection(
                    candidate_to_types.get(c.concept_id, []),
                    list(PREFERRED_UMLS_TYPES.keys()),
                )
            ]
            return preferred_type_candidates

        def select_members():
            real_candidates = [c for c in candidates if c.similarities[0] >= 0]

            # Partial match if non-matched words, and only a single candidate (TODO: revisit)
            is_partial = (
                # has 1+ fake candidates (i.e. unmatched)
                sum(c.similarities[0] < 0 for c in candidates) > 0
                # and only one real candidate match
                and len(real_candidates) == 1
                # and that candidate is a single word (imperfect proxy for it being a 1gram match)
                and len(real_candidates[0].aliases[0].split(" ")) == 1
            )

            # if partial match, include *all* candidates, which includes the faked ones
            # "UNMATCHED inhibitor" will have a name and id that reflects the unmatched word
            if is_partial:
                return candidates

            # else, we're going to drop unmatched words
            # e.g. "cpla (2)-selective inhibitor" -> "cpla inhibitor"

            # if we have 1+ preferred candidates, return those
            # this prevents composites like C0024579|C0441833 ("Maleimides Groups") - wherein "Group" offers little value
            preferred = get_preferred_typed_candidates(real_candidates)
            if len(preferred) >= 1:
                return preferred

            return real_candidates

        member_candidates = select_members()

        if len(member_candidates) == 0:
            return None

        # if just a single composite match, treat it like a non-composite match
        if len(member_candidates) == 1:
            return self._candidate_to_canonical(member_candidates[0])

        # sorted for the sake of consist composite ids
        ids = sorted([c.concept_id for c in member_candidates])

        # form name from comprising candidates
        name = self._form_composite_name(member_candidates)

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=name,
            # description=..., # TODO: composite description
            # aliases=... # TODO: all permutations
        )

    def _generate_composite(
        self,
        mention_text: str,
        ngram_candidate_map: Mapping[str, MentionCandidate],
    ) -> CanonicalEntity | None:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            ngram_candidate_map (dict[str, MentionCandidate]): word-to-candidate map
        """
        if mention_text.strip() == "":
            return None

        def get_composite_candidates(words: tuple[str, ...]) -> list[MentionCandidate]:
            """
            Recursive function to see if the first ngram has a match, then the first n-1, etc.
            """
            if len(words) == 0:
                return []

            if len(words) >= NGRAMS_N:
                ngram = " ".join(words[0:NGRAMS_N])
                if ngram in ngram_candidate_map:
                    remaining_words = tuple(words[NGRAMS_N:])
                    return [
                        ngram_candidate_map[ngram],
                        *get_composite_candidates(remaining_words),
                    ]

            # otherwise, let's map only the first word
            remaining_words = tuple(words[1:])
            if words[0] in ngram_candidate_map:
                return [
                    ngram_candidate_map[words[0]],
                    *get_composite_candidates(remaining_words),
                ]

            # otherwise, no match. create a fake MentionCandidate.
            return [
                # concept_id is the word itself, so
                # composite id will look like "UNMATCHED|C1999216" for "UNMATCHED inhibitor"
                MentionCandidate(
                    concept_id=words[0].lower(),
                    aliases=[words[0]],
                    similarities=[-1],
                ),
                *get_composite_candidates(remaining_words),
            ]

        all_words = self._get_words(mention_text)
        candidates = get_composite_candidates(all_words)

        return self._form_composite(candidates)

    # def _optimize_composites(
    #     self, composite_matches: dict[str, CanonicalEntity | None]
    # ) -> dict[str, CanonicalEntity | None]:
    #     """
    #     Taking the new composite names, see if there is now a singular match
    #     (e.g. a composite name might be "SGLT2 inhibitor", comprised of two candidates, for which a single match exists)
    #     """
    #     composite_names = uniq(
    #         [cm.name for cm in composite_matches.values() if cm is not None]
    #     )
    #     direct_match_map = {
    #         n: self._get_best_canonical(c)
    #         for n, c in zip(composite_names, self._get_candidates(composite_names))
    #     }
    #     # combine composite and potential single matches
    #     return {
    #         t: (direct_match_map.get(cm.name) if cm is not None else cm) or cm
    #         for t, cm in composite_matches.items()
    #     }

    # def _generate_composite_entities(
    #     self, matchless_mention_texts: Sequence[str]
    # ) -> dict[str, CanonicalEntity]:
    #     """
    #     For a list of mention text without a sufficiently similar direct match,
    #     generate a composite match from the individual words

    #     Args:
    #         matchless_mention_texts (Sequence[str]): list of mention texts
    #     """

    #     # create 1 and 2grams
    #     matchless_ngrams = uniq(
    #         flatten(
    #             [
    #                 self._get_ngrams(text, i + 1)
    #                 for text in matchless_mention_texts
    #                 for i in range(NGRAMS_N)
    #             ]
    #         )
    #     )

    #     # get candidates for all ngrams
    #     matchless_candidates = self._get_candidates(matchless_ngrams)

    #     # create a map of ngrams to the best candidate
    #     ngram_candidate_map: dict[str, MentionCandidate] = omit_by(
    #         {
    #             ngram: self.get_best_candidate(candidate_set)
    #             for ngram, candidate_set in zip(matchless_ngrams, matchless_candidates)
    #         },
    #         lambda v: v is None,
    #     )

    #     # generate the composites
    #     composite_matches = {
    #         mention_text: self._generate_composite(mention_text, ngram_candidate_map)
    #         for mention_text in matchless_mention_texts
    #     }

    #     # "optimize" the matches by passing the composite names back through the candidate generator
    #     # to see if there is now a direct match
    #     optimized_matches = self._optimize_composites(composite_matches)

    #     return omit_by(optimized_matches, lambda v: v is None)

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = self._get_candidates(entity.term)

        match = self._get_best_canonical(candidates, entity)

        # matchless = self._generate_composite_entities(
        #     [
        #         text
        #         for text, canonical in matches.items()
        #         if not CompositeCandidateSelector._is_composite_eligible(
        #             text, canonical
        #         )
        #     ],
        # )

        # # combine composite matches such that they override the original matches
        # all_matches: dict[str, CanonicalEntity | None] = {**matches, **matchless}

        return match
