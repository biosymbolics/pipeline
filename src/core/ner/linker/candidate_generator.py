from typing import Mapping, Sequence
from pydash import flatten, omit_by, uniq
from spacy.lang.en.stop_words import STOP_WORDS
from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
)

from core.ner.types import CanonicalEntity
from utils.string import generate_ngram_phrases


MIN_WORD_LENGTH = 1
NGRAMS_N = 2
DEFAULT_K = 3  # mostly wanting to avoid suppressions. increase if adding a lot more suppressions.

CUI_SUPPRESSIONS = {
    "C0432616": "Blood group antibody A",  # matches "anti", sigh
    "C1413336": "CEL gene",  # matches "cell"; TODO fix so this gene can match
    "C1413568": "COIL gene",  # matches "coil"
    "C0439095": "greek letter alpha",
    "C0439096": "greek letter beta",
    "C0439097": "greek letter delta",
    "C1552644": "greek letter gamma",
    "C0231491": "antagonist muscle action",  # blocks better match (C4721408)
    "C0332281": "Associated with",
}

# TODO: maybe choose NCI as canonical name
CANONICAL_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
}


class CompositeCandidateGenerator(CandidateGenerator, object):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(canonical_id, '\\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;

    - Certain gene names are matched naively (e.g. "cell" -> CEL gene, tho that one in particular is suppressed)
    - dedup same ids, e.g. Keratin fiber / Keratin fibre both C0010803|C0225326 - combine
    - potentially suppress some types as means to "this is not substantive"
        - T077 residue, means, cure
        - T090 (occupation) Technology, engineering, Magnetic <?>
        - T079 (Temporal Concept) date, future
        - T078 (idea or concept) INFORMATION, bias, group
        - T080 (Qualitative Concept) includes solid, biomass
        - T081 (Quantitative Concept) includes Bioavailability, bacterial
        - T082 spatial - includes bodily locations like 'Prostatic', terms like Occlusion, Polycyclic, lateral
        - T169 functional - includes ROAs and endogenous/exogenous, but still probably okay to remove
        - T041 mental process - e.g. "like" (as in, "I like X")
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

    @classmethod
    def get_best_candidate(
        cls, candidates: Sequence[MentionCandidate], min_similarity: float
    ) -> MentionCandidate | None:
        """
        Finds the best candidate

        - Sufficient similarity
        - Not suppressed
        """
        ok_candidates = sorted(
            [
                c
                for c in candidates
                if c.similarities[0] >= min_similarity
                and c.concept_id not in CUI_SUPPRESSIONS
            ],
            key=lambda c: c.similarities[0],
            reverse=True,
        )

        return ok_candidates[0] if len(ok_candidates) > 0 else None

    def _create_composite_name(self, candidates: Sequence[MentionCandidate]) -> str:
        """
        Create a composite name from a list of candidates
        (all canonical names, concatted)
        """
        return " ".join(
            [
                self.kb.cui_to_entity[c.concept_id].canonical_name
                if c.concept_id in self.kb.cui_to_entity
                else c.aliases[0]
                for c in candidates
            ]
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

        def get_candidates(words: tuple[str, ...]) -> list[MentionCandidate]:
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
                        *get_candidates(remaining_words),
                    ]

            # otherwise, let's map only the first word
            remaining_words = tuple(words[1:])
            if words[0] in ngram_candidate_map:
                return [ngram_candidate_map[words[0]], *get_candidates(remaining_words)]

            return [
                MentionCandidate(
                    concept_id="na",
                    aliases=[words[0]],
                    similarities=[-1],
                ),
                *get_candidates(remaining_words),
            ]

        all_words = self.get_words(mention_text)
        candidates = get_candidates(all_words)

        ids = sorted([c.concept_id for c in candidates if c.similarities[0] > 0])

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=self._create_composite_name(candidates),
            # aliases=... # TODO: all permutations
        )

    def generate_composite_entities(
        self, matchless_mention_texts: Sequence[str], min_similarity: float
    ) -> dict[str, CanonicalEntity]:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_mention_texts (Sequence[str]): list of mention texts
            min_similarity (float): minimum similarity to consider a match
        """

        # 1 and 2grams
        matchless_ngrams = uniq(
            flatten(
                [
                    self.get_ngrams(text, i + 1)
                    for text in matchless_mention_texts
                    for i in range(NGRAMS_N)
                ]
            )
        )

        # get candidates from superclass
        matchless_candidates = super().__call__(matchless_ngrams, k=DEFAULT_K)

        # create a map of ngrams to (acceptable) candidates
        ngram_candidate_map: dict[str, MentionCandidate] = omit_by(
            {
                ngram: self.get_best_candidate(candidate_set, self.min_similarity)
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

    def _get_canonical(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity

        Args:
            candidates (Sequence[MentionCandidate]): candidates
        """
        top_candidate = self.get_best_candidate(candidates, self.min_similarity)

        if top_candidate is None:
            return None

        # go to kb to get canonical name
        entity = self.kb.cui_to_entity[candidates[0].concept_id]

        return CanonicalEntity(
            id=entity.concept_id,
            ids=[entity.concept_id],
            name=entity.canonical_name,
            aliases=entity.aliases,
            description=entity.definition,
            types=entity.types,
        )

    def __call__(self, mention_texts: Sequence[str]) -> list[CanonicalEntity]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = super().__call__(list(mention_texts), k=DEFAULT_K)

        matches = {
            mention_text: self._get_canonical(candidate_set)
            for mention_text, candidate_set in zip(mention_texts, candidates)
        }

        matchless = self.generate_composite_entities(
            [text for text, canonical in matches.items() if canonical is None],
            self.min_similarity,
        )

        # combine composite matches such that they override the original matches
        all_matches: dict[str, CanonicalEntity] = {**matches, **matchless}  # type: ignore

        # ensure order
        return [all_matches[mention_text] for mention_text in mention_texts]
