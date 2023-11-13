from typing import Mapping, Sequence
from pydash import flatten, omit_by, uniq
from spacy.lang.en.stop_words import STOP_WORDS
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate

from core.ner.types import CanonicalEntity
from data.common.biomedical.umls import clean_umls_name, get_best_umls_candidate
from utils.string import generate_ngram_phrases

MIN_WORD_LENGTH = 1
NGRAMS_N = 2
DEFAULT_K = 3  # mostly wanting to avoid suppressions. increase if adding a lot more suppressions.

CANDIDATE_CUI_SUPPRESSIONS = {
    "C0432616": "Blood group antibody A",  # matches "anti", sigh
    "C1413336": "CEL gene",  # matches "cell"; TODO fix so this gene can match
    "C1413568": "COIL gene",  # matches "coil"
    "C0231491": "antagonist muscle action",  # blocks better match (C4721408)
    "C0205263": "Induce (action)",
    "C1709060": "Modulator device",
    "C0179302": "Binder device",
    "C0280041": "Substituted Urea",  # matches all "substituted" terms, sigh
}


# map term to specified cui
COMPOSITE_WORD_OVERRIDES = {
    "modulator": "C0005525",  # "Biological Response Modifiers"
    "modulators": "C0005525",
    "binder": "C1145667",  # "Binding action"
    "binders": "C1145667",
}

# pde-v inhibitor  - works of pde-v but not pde v or pdev
# bace 2 inhibitor - base2
# glp-2 agonist - works with dash


class CompositeCandidateGenerator(CandidateGenerator, object):
    """
    A candidate generator that if not finding a suitable candidate, returns a composite candidate

    Look up in UMLS here and use 'type' for standard ordering on composite candidate names (e.g. gene first)
    select  s.term, array_agg(type_name), array_agg(type_id), ids from (select term, regexp_split_to_array(id, '\\|') ids from terms) s, umls_lookup, unnest(s.ids) as idd  where idd=umls_lookup.id and array_length(ids, 1) > 1 group by s.term, ids;

    - Certain gene names are matched naively (e.g. "cell" -> CEL gene, tho that one in particular is suppressed)
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
        Finds the best candidate
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

    def _get_matches(self, texts: Sequence[str]) -> list[list[MentionCandidate]]:
        """
        Wrapper around super().__call__ that handles word overrides
        """
        candidates = super().__call__(list(texts), k=DEFAULT_K)
        return self._apply_word_overrides(texts, candidates)

    def _create_composite_name(self, candidates: Sequence[MentionCandidate]) -> str:
        """
        Create a composite name from a list of candidates
        """

        def get_name(c):
            if c.concept_id in self.kb.cui_to_entity:
                ce = self.kb.cui_to_entity[c.concept_id]
                return clean_umls_name(ce.concept_id, ce.canonical_name, ce.aliases)

            return c.aliases[0]

        return " ".join([get_name(c) for c in candidates])

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
        composite_name = self._create_composite_name(candidates)

        # TODO!!! tfidf check - is sufficiently close to manufactured term?

        return CanonicalEntity(
            id="|".join(ids),
            ids=ids,
            name=composite_name,
            # description=..., # TODO: composite description
            # aliases=... # TODO: all permutations
        )

    def generate_composite_entities(
        self, matchless_mention_texts: Sequence[str]
    ) -> dict[str, CanonicalEntity]:
        """
        For a list of mention text without a sufficiently similar direct match,
        generate a composite match from the individual words

        Args:
            matchless_mention_texts (Sequence[str]): list of mention texts
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
        matchless_candidates = self._get_matches(matchless_ngrams)

        # create a map of ngrams to (acceptable) candidates
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

    def _get_canonical(
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

        # go to kb to get canonical name
        entity = self.kb.cui_to_entity[top_candidate.concept_id]
        name = clean_umls_name(entity.concept_id, entity.canonical_name, entity.aliases)

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
        candidates = self._get_matches(mention_texts)

        matches = {
            mention_text: self._get_canonical(candidate_set)
            for mention_text, candidate_set in zip(mention_texts, candidates)
        }

        matchless = self.generate_composite_entities(
            [text for text, canonical in matches.items() if canonical is None],
        )

        # combine composite matches such that they override the original matches
        all_matches: dict[str, CanonicalEntity] = {**matches, **matchless}  # type: ignore

        # ensure order
        return [all_matches[mention_text] for mention_text in mention_texts]
