from typing import Mapping, Sequence
from pydash import flatten, omit_by, uniq
from spacy.lang.en.stop_words import STOP_WORDS
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
from constants.patterns.iupac import is_iupac

from core.ner.types import CanonicalEntity
from constants.umls import PREFERRED_UMLS_TYPES, UMLS_CUI_SUPPRESSIONS
from data.domain.biomedical.umls import clean_umls_name, get_best_umls_candidate
from utils.list import has_intersection
from utils.string import generate_ngram_phrases

MIN_WORD_LENGTH = 1
NGRAMS_N = 2
DEFAULT_K = 3  # mostly wanting to avoid suppressions. increase if adding a lot more suppressions.

CANDIDATE_CUI_SUPPRESSIONS = {
    **UMLS_CUI_SUPPRESSIONS,
    "C0432616": "Blood group antibody A",  # matches "anti", sigh
    "C1704653": "cell device",  # matches "cell"
    "C0231491": "antagonist muscle action",  # blocks better match (C4721408)
    "C0205263": "Induce (action)",
    "C1709060": "Modulator device",
    "C0179302": "Binder device",
    "C0280041": "Substituted Urea",  # matches all "substituted" terms, sigh
    "C1179435": "Protein Component",  # sigh... matches "component"
    "C0870814": "like",
    "C0080151": "Simian Acquired Immunodeficiency Syndrome",  # matches "said"
    "C0163712": "Relate - vinyl resin",
    "C2827757": "Antimicrobial Resistance Result",  # ("result") ugh
    "C1882953": "ring",
    "C0457385": "seconds",  # s
    "C0179636": "cart",  # car-t
    "C0039552": "terminally ill",
    "C0175816": "https://uts.nlm.nih.gov/uts/umls/concept/C0175816",
    "C0243072": "derivative",
    "C1744692": "NOS inhibitor",  # matches "inhibitor"
}


# map term to specified cui
COMPOSITE_WORD_OVERRIDES = {
    "modulator": "C0005525",  # "Biological Response Modifiers"
    "modulators": "C0005525",
    "binder": "C1145667",  # "Binding action"
    "binders": "C1145667",
}


class CompositeCandidateGenerator(CandidateGenerator, object):
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

    def __init__(self, *args, min_similarity: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

    @classmethod
    def get_words(cls, text: str) -> tuple[str, ...]:
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
    def get_ngrams(cls, text: str, n: int) -> list[str]:
        """
        Get all ngrams in a text
        """
        words = cls.get_words(text)

        # if fewer words than n, just return words
        # (this is expedient but probably confusing)
        if n == 1 or len(words) < n:
            return list(words)

        ngrams = generate_ngram_phrases(words, n)
        return ngrams

    def get_best_candidate(
        self, candidates: Sequence[MentionCandidate]
    ) -> MentionCandidate | None:
        """
        Wrapper for get_best_umls_candidate
        """

        return get_best_umls_candidate(
            candidates,
            self.min_similarity,
            self.kb,
            list(CANDIDATE_CUI_SUPPRESSIONS.keys()),
        )

    @classmethod
    def _apply_word_overrides(
        cls, texts: Sequence[str], candidates: list[list[MentionCandidate]]
    ) -> list[list[MentionCandidate]]:
        """
        Certain words we match to an explicit cui (e.g. "modulator" -> "C0005525")
        """
        # look for any overrides (terms -> candidate)
        override_indices = [
            i for i, t in enumerate(texts) if t.lower() in COMPOSITE_WORD_OVERRIDES
        ]
        for i in override_indices:
            candidates[i] = [
                MentionCandidate(
                    concept_id=COMPOSITE_WORD_OVERRIDES[texts[i].lower()],
                    aliases=[texts[i]],
                    similarities=[1],
                )
            ]
        return candidates

    def _get_candidates(self, texts: Sequence[str]) -> list[list[MentionCandidate]]:
        """
        Wrapper around super().__call__ that handles word overrides
        """
        candidates = super().__call__(list(texts), k=DEFAULT_K)
        with_overrides = self._apply_word_overrides(texts, candidates)
        return with_overrides

    def _assemble_composite(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Form a composite from a list of candidates
        """

        def get_name_part(c):
            if c.concept_id in self.kb.cui_to_entity:
                ce = self.kb.cui_to_entity[c.concept_id]
                return clean_umls_name(
                    ce.concept_id, ce.canonical_name, ce.aliases, True
                )
            return c.aliases[0]

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

        def get_member_candidates():
            real_candidates = [c for c in candidates if c.similarities[0] >= 0]

            # Partial match if non-matched words, and only a single candidate (TODO: revisit)
            is_partial = (
                # has 1+ fake candidates (i.e. unmatched)
                sum(c.similarities[0] < 0 for c in candidates) > 0
                # and only one real candidate match
                and len(real_candidates) == 1
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

        member_candidates = get_member_candidates()

        if len(member_candidates) == 0:
            return None

        # if just a single composite match, treat it like a non-composite match
        if len(member_candidates) == 1:
            return self._candidate_to_canonical(member_candidates[0])

        # sorted for the sake of consist composite ids
        ids = sorted([c.concept_id for c in member_candidates])
        name = " ".join([get_name_part(c) for c in member_candidates])

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

        all_words = self.get_words(mention_text)
        candidates = get_composite_candidates(all_words)

        return self._assemble_composite(candidates)

    def generate_composite_entities(
        self, matchless_mention_texts: Sequence[str]
    ) -> dict[str, CanonicalEntity]:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_mention_texts (Sequence[str]): list of mention texts
        """

        # create 1 and 2grams
        matchless_ngrams = uniq(
            flatten(
                [
                    self.get_ngrams(text, i + 1)
                    for text in matchless_mention_texts
                    for i in range(NGRAMS_N)
                ]
            )
        )

        # get candidates for all ngrams
        matchless_candidates = self._get_candidates(matchless_ngrams)

        # create a map of ngrams to the best candidate
        ngram_candidate_map: dict[str, MentionCandidate] = omit_by(
            {
                ngram: self.get_best_candidate(candidate_set)
                for ngram, candidate_set in zip(matchless_ngrams, matchless_candidates)
            },
            lambda v: v is None,
        )

        # generate the composites
        composite_matches = {
            mention_text: self._generate_composite(mention_text, ngram_candidate_map)
            for mention_text in matchless_mention_texts
        }

        return {t: m for t, m in composite_matches.items() if m is not None}

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

    def _get_best_canonical(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity

        Args:
            candidates (Sequence[MentionCandidate]): candidates
        """
        top_candidate = self.get_best_candidate(candidates)

        if top_candidate is None:
            return None

        return self._candidate_to_canonical(top_candidate)

    def _candidate_to_canonical(self, candidate: MentionCandidate) -> CanonicalEntity:
        """
        Convert a MentionCandidate to a CanonicalEntity
        """
        # go to kb to get canonical name
        entity = self.kb.cui_to_entity[candidate.concept_id]
        name = clean_umls_name(
            entity.concept_id, entity.canonical_name, entity.aliases, False
        )

        return CanonicalEntity(
            id=entity.concept_id,
            ids=[entity.concept_id],
            name=name,
            aliases=entity.aliases,
            description=entity.definition,
            types=entity.types,
        )

    def __call__(self, mention_texts: Sequence[str]) -> list[CanonicalEntity]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = self._get_candidates(mention_texts)

        matches = {
            mention_text: self._get_best_canonical(candidate_set)
            for mention_text, candidate_set in zip(mention_texts, candidates)
        }

        matchless = self.generate_composite_entities(
            [
                text
                for text, canonical in matches.items()
                if not CompositeCandidateGenerator._is_composite_eligible(
                    text, canonical
                )
            ],
        )

        # combine composite matches such that they override the original matches
        all_matches: dict[str, CanonicalEntity] = {**matches, **matchless}  # type: ignore

        # ensure order
        return [all_matches[mention_text] for mention_text in mention_texts]
