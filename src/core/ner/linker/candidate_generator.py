from typing import Sequence
from pydash import flatten, mean, uniq

from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
from spacy.lang.en.stop_words import STOP_WORDS

from utils.string import generate_ngram_phrases

MIN_WORD_LENGTH = 1
NGRAMS_N = 2

CUI_SUPPRESSIONS = {
    "C0432616": "Blood group antibody A",  # matches "anti", sigh
    "C1413336": "CEL gene",  # matches "cell"; TODO fix so this gene can match
    "C1413568": "COIL gene",  # matches "coil"
    "C0439095": "greek letter alpha",
}


class CompositeCandidateGenerator(CandidateGenerator, object):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(canonical_id, '\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;

    - Certain gene names are matched naively (e.g. "cell" -> CEL gene, tho that one in particular is suppressed)
    - not first alias (https://uts.nlm.nih.gov/uts/umls/concept/C0127400)
    - dedup same ids, e.g. Keratin fiber / Keratin fibre both C0010803|C0225326 - combine
    - potentially suppress some types
        - T077 residue, means, cure
        - T090 (occupation) Technology, engineering, Magnetic <?>
        - T079 (Temporal Concept) date, future
        - T078 (idea or concept) INFORMATION, bias, group
        - T080 (Qualitative Concept) includes solid, biomass
        - T081 (Quantitative Concept) includes Bioavailability, bacterial
        - T082 spatial - includes bodily locations like 'Prostatic', terms like Occlusion, Polycyclic, lateral
        - T169 functional - includes ROAs and endogenous/exogenous, but still probably okay to remove
        - T041 mental process - may have a few false positives

    - look at microglia
    """

    def __init__(self, *args, min_similarity: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

    @classmethod
    def get_words(cls, text: str) -> tuple[str, ...]:
        return tuple(
            [
                word
                for word in text.split()
                if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS
            ]
        )

    @classmethod
    def get_ngrams(cls, text: str, n: int) -> list[str]:
        words = cls.get_words(text)

        # if fewer words than n, just return words
        # (this is expedient but probably confusing)
        if n == 1 or len(words) < n:
            return list(words)

        ngrams = generate_ngram_phrases(words, n)
        return ngrams

    @classmethod
    def is_ok_candidate(
        cls, candidates: Sequence[MentionCandidate], min_similarity: float
    ) -> bool:
        # k = 1 so each should have only 1 entry anyway
        return (
            len(candidates) > 0
            and len(candidates[0].similarities) > 0
            and candidates[0].similarities[0] > min_similarity
            and candidates[0].concept_id not in CUI_SUPPRESSIONS
        )

    @classmethod
    def _generate_composite(
        cls,
        mention_text: str,
        ngram_candidate_map: dict[str, MentionCandidate],
    ) -> list[MentionCandidate]:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            ngram_candidate_map (dict[str, MentionCandidate]): word-to-candidate map
        """
        if mention_text.strip() == "":
            return []

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

        all_words = cls.get_words(mention_text)
        candidates = get_candidates(all_words)

        # similarity score as mean of all words
        similarities = [c.similarities[0] for c in candidates if c.similarities[0] > 0]
        similarity = mean(similarities) if len(similarities) > 0 else -1

        return [
            MentionCandidate(
                concept_id="|".join(
                    [c.concept_id for c in candidates if c.similarities[0] > 0]
                ),
                # TODO: all permutations
                aliases=[" ".join([c.aliases[0] for c in candidates])],
                similarities=[similarity],
            )
        ]

    def find_composite_matches(
        self, matchless_mention_texts: Sequence[str], min_similarity: float
    ) -> dict[str, list[MentionCandidate]]:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_mention_texts (Sequence[str]): list of mention texts
            min_similarity (float): minimum similarity to consider a match
        """

        # 1grams and 2grams
        matchless_ngrams = uniq(
            flatten(
                [
                    self.get_ngrams(text, i + 1)
                    for text in matchless_mention_texts
                    for i in range(NGRAMS_N)
                ]
            )
        )

        matchless_candidates = super().__call__(matchless_ngrams, 1)
        ngram_candidate_map = {
            ngram: candidate[0]
            for ngram, candidate in zip(matchless_ngrams, matchless_candidates)
            if self.is_ok_candidate(candidate, min_similarity)
        }
        composite_matches = {
            mention_text: self._generate_composite(mention_text, ngram_candidate_map)
            for mention_text in matchless_mention_texts
        }
        return composite_matches

    def __call__(
        self, mention_texts: Sequence[str], k: int
    ) -> list[list[MentionCandidate]]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = super().__call__(list(mention_texts), k)

        matches = {
            mention_text: candidate
            for mention_text, candidate in zip(mention_texts, candidates)
        }

        # combine composite matches such that they override the original matches
        all_matches = {
            **matches,
            **self.find_composite_matches(
                [
                    mention_text
                    for mention_text in mention_texts
                    if not self.is_ok_candidate(
                        matches[mention_text], self.min_similarity
                    )
                ],
                min_similarity=self.min_similarity,
            ),
        }

        # ensure order
        return [all_matches[mention_text] for mention_text in mention_texts]
