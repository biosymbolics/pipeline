from typing import Sequence
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate

from core.ner.types import CanonicalEntity
from constants.umls import UMLS_CUI_SUPPRESSIONS
from data.domain.biomedical.umls import clean_umls_name, get_best_umls_candidate

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


class CandidateSelector(CandidateGenerator, object):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    """

    def __init__(self, *args, min_similarity: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity

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

    def _get_candidates(
        self, mention_texts: Sequence[str]
    ) -> list[list[MentionCandidate]]:
        """
        Wrapper around super().__call__ that handles word overrides
        """
        candidates = super().__call__(list(mention_texts), k=DEFAULT_K)
        with_overrides = self._apply_word_overrides(mention_texts, candidates)
        return with_overrides

    def _get_best_canonical(
        self, candidates: Sequence[MentionCandidate]
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity
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
            entity.concept_id,
            entity.canonical_name,
            entity.aliases,
            entity.types,
            False,
        )

        return CanonicalEntity(
            id=entity.concept_id,
            ids=[entity.concept_id],
            name=name,
            aliases=entity.aliases,
            description=entity.definition,
            types=entity.types,
        )

    def __call__(self, mention_texts: Sequence[str]) -> list[CanonicalEntity | None]:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = self._get_candidates(mention_texts)

        matches = {
            mention_text: self._get_best_canonical(candidate_set)
            for mention_text, candidate_set in zip(mention_texts, candidates)
        }

        # ensure order
        return [matches[mention_text] for mention_text in mention_texts]
