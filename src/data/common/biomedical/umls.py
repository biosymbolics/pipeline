from typing import Sequence
from scispacy.candidate_generation import MentionCandidate

from utils.list import has_intersection
from .constants import (
    BIOMEDICAL_UMLS_TYPES,
    UMLS_NAME_OVERRIDES,
    UMLS_NAME_SUPPRESSIONS,
)


def get_best_umls_candidate(
    candidates: Sequence[MentionCandidate],
    min_similarity: float,
    kb,
    cui_suppressions: list[str] = [],
) -> MentionCandidate | None:
    """
    Finds the best candidate between "MentionCandidates" (produced by SciSpacy)

    - Sufficient similarity
    - Not suppressed

    Args:
        candidates (Sequence[MentionCandidate]): candidates
        cui_suppressions (list[str], optional): suppressed cuis. Defaults to [].
    """

    def sorter(c: MentionCandidate):
        types = set(kb.cui_to_entity[c.concept_id].types)

        # sort non-preferred-types to the bottom
        if not has_intersection(types, BIOMEDICAL_UMLS_TYPES):
            return min_similarity

        return c.similarities[0]

    ok_candidates = sorted(
        [
            c
            for c in candidates
            if c.similarities[0] >= min_similarity
            and c.concept_id not in cui_suppressions
            and not all(
                [
                    t in BIOMEDICAL_UMLS_TYPES
                    for t in kb.cui_to_entity[c.concept_id].types
                ]
            )
            and not has_intersection(
                UMLS_NAME_SUPPRESSIONS,
                kb.cui_to_entity[c.concept_id].canonical_name.split(" "),
            )
        ],
        key=sorter,
        reverse=True,
    )

    return ok_candidates[0] if len(ok_candidates) > 0 else None


def clean_umls_name(cui: str, canonical_name: str, aliases: list[str]) -> str:
    """
    Cleans up UMLS names, potentially choosing an alias over the canonical name

    - prefer shorter names
    - prefer names that are `XYZ protein` vs `protein, XYZ`
    - prefer names that start with the same word/letter as the canonical name
    """
    if cui in UMLS_NAME_OVERRIDES:
        return UMLS_NAME_OVERRIDES[cui]

    name_words = canonical_name.split(" ")

    def name_sorter(a: str) -> int:
        # prefer shorter aliases that start with the same word/letter as the canonical name
        # e.g. TNFSF11 for "TNFSF11 protein, human"
        # todo: use something like tfidf

        return (
            len(a)
            # prefer same first word
            + (5 if a.split(" ")[0].lower() != name_words[0].lower() else 0)
            # prefer same first letter
            + (20 if a[0].lower() != canonical_name[0].lower() else 0)
            # prefer non-comma
            + (5 if "," in a else 0)
        )

    # if 1-2 words or no aliases, prefer canonical name
    if len(name_words) <= 2 or len(aliases) == 0:
        return canonical_name

    aliases = sorted(aliases, key=name_sorter)
    return aliases[0]
