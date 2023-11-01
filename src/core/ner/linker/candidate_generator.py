from typing import Sequence
from pydash import mean, uniq

from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
from spacy.lang.en.stop_words import STOP_WORDS


class CompositeCandidateGenerator(CandidateGenerator, object):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    TODO: look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    """

    def __init__(self, *args, min_similarity: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

    def find_composite_matches(
        self, matchless_mention_texts: Sequence[str], min_similarity: float
    ):
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_mention_texts (Sequence[str]): list of mention texts
            min_similarity (float): minimum similarity to consider a match
        """
        matchless_words = uniq(
            [
                word
                for text in matchless_mention_texts
                for word in text.split()
                if len(word) > 2 and word not in STOP_WORDS
            ]
        )
        matchless_candidates = super().__call__(matchless_words, 1)
        word_candidate_map = {
            # k = 1 so each should have only 1 entry anyway
            word: candidate[0]
            if len(candidate) > 0
            and len(candidate[0].similarities) > 0
            and candidate[0].similarities[0] > min_similarity
            else MentionCandidate("na", [word], [1])
            for word, candidate in zip(matchless_words, matchless_candidates)
        }
        composite_matches = {
            mention_text: CompositeCandidateGenerator._generate_composite(
                mention_text, word_candidate_map
            )
            for mention_text in matchless_mention_texts
        }
        return composite_matches

    @staticmethod
    def _generate_composite(
        mention_text: str,
        word_candidate_map: dict[str, MentionCandidate],
    ) -> Sequence[MentionCandidate]:
        """
        Generate a composite candidate from a mention text

        Args:
            mention_text (str): Mention text
            word_candidate_map (dict[str, MentionCandidate]): word-to-candidate map
        """
        if mention_text.strip() == "":
            return []

        candidates = [
            word_candidate_map.get(word)
            # fake candidate, just to make assembly easier
            or MentionCandidate(
                concept_id="na",
                aliases=[word],
                similarities=[-1],
            )
            for word in mention_text.split()
        ]

        # similarity score as mean of all words
        similarity = mean(
            [c.similarities[0] for c in candidates if c.similarities[0] > 0]
        )

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
