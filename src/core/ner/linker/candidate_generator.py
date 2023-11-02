from typing import Sequence
from pydash import flatten, mean, uniq

from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
from spacy.lang.en.stop_words import STOP_WORDS

from utils.string import generate_ngram_phrases

MIN_WORD_LENGTH = 2


class CompositeCandidateGenerator(CandidateGenerator, object):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(canonical_id, '\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;
    - Grabby
      - Antibodies too grabby, e.g. "antibody against c5"
      - Antibodies, Anti-Idiotypic -> anti- αβ42 antibody, anti-tnfalpha antibody
      - Antagonist muscle action -> ανβ3 receptor antagonists etc
      - "anti-cancer therapeutics" -> Anti A Cancer
    - Wrong
      - C1413336 / CEL gene - matches cell!!
      - Emitter COIL - COIL gene
      - FLAME gene
      - https://uts.nlm.nih.gov/uts/umls/concept/C0870814 - "like"
    - not first alias (https://uts.nlm.nih.gov/uts/umls/concept/C0127400)
    - dedup same ids, e.g. Keratin fiber / Keratin fibre both C0010803|C0225326 - combine
    - bigrams
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

        - remove short text? (e.g., food Content
        - leave term if only umls match



    - look at microglia
    """

    def __init__(self, *args, min_similarity: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

    @classmethod
    def get_ngrams(cls, texts: Sequence[str], n: int = 1) -> list[str]:
        def _ngram(text: str, n: int) -> list[str]:
            words = [
                word
                for word in text.split()
                if len(word) > MIN_WORD_LENGTH and word not in STOP_WORDS
            ]

            # if fewer words than n, just return words
            # (this is expedient but probably confusing)
            if n == 1 or len(words) < n:
                return words

            ngrams = generate_ngram_phrases(words, n)
            return ngrams

        matchless_ngrams = uniq(flatten([_ngram(text, n) for text in texts]))
        return matchless_ngrams

    @classmethod
    def _generate_composite(
        cls,
        mention_text: str,
        word_candidate_map: dict[str, MentionCandidate],
    ) -> list[MentionCandidate]:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            word_candidate_map (dict[str, MentionCandidate]): word-to-candidate map
        """
        if mention_text.strip() == "":
            return []

        ngrams = cls.get_ngrams([mention_text], 2)

        def get_ngram_candidate(ngram: str, is_last: bool) -> list[MentionCandidate]:
            """
            Return candidate for a ngram
            (if no ngram candidate, return a candidate for each word in the ngram)
            """
            if ngram in word_candidate_map:
                return [word_candidate_map[ngram]]

            single_words = ngram.split(" ")[-1 if is_last else 0]

            return [
                word_candidate_map[word]
                if word in word_candidate_map
                else MentionCandidate(
                    concept_id="na",
                    aliases=[word],
                    similarities=[-1],
                )
                for word in single_words
            ]

        candidates = flatten(
            [
                get_ngram_candidate(ngram, is_last=(i == len(ngrams) - 1))
                for i, ngram in enumerate(ngrams)
            ]
        )

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

        def has_acceptable_candidate(candidates: Sequence[MentionCandidate]) -> bool:
            return (
                len(candidates) > 0
                and len(candidates[0].similarities) > 0
                and candidates[0].similarities[0] > min_similarity
            )

        # 1grams and 2grams
        matchless_ngrams = flatten(
            [self.get_ngrams(matchless_mention_texts, i) for i in range(1, 2)]
        )

        matchless_candidates = super().__call__(matchless_ngrams, 1)
        word_candidate_map = {
            # k = 1 so each should have only 1 entry anyway
            word: candidate[0]
            if has_acceptable_candidate(candidate)
            else MentionCandidate("na", [word], [1])
            for word, candidate in zip(matchless_ngrams, matchless_candidates)
        }
        composite_matches = {
            mention_text: self._generate_composite(mention_text, word_candidate_map)
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

        def has_match(candidates: Sequence[MentionCandidate]) -> bool:
            return (
                len(candidates) > 0
                and candidates[0].similarities[0] > self.min_similarity
            )

        # combine composite matches such that they override the original matches
        all_matches = {
            **matches,
            **self.find_composite_matches(
                [
                    mention_text
                    for mention_text in mention_texts
                    if not has_match(matches[mention_text])
                ],
                min_similarity=self.min_similarity,
            ),
        }

        # ensure order
        return [all_matches[mention_text] for mention_text in mention_texts]
