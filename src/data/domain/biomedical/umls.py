from typing import Sequence
from scispacy.candidate_generation import MentionCandidate

from constants.umls import (
    UMLS_GENE_PROTEIN_TYPES,
    UMLS_NAME_OVERRIDES,
    UMLS_NAME_SUPPRESSIONS,
    PREFERRED_UMLS_TYPES,
    MOST_PREFERRED_UMLS_TYPES,
)
from utils.list import has_intersection


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
        min_similarity (float): minimum similarity
        kb ([KnowledgeBase]): knowledge base (SciSpacy)
        cui_suppressions (list[str], optional): suppressed cuis. Defaults to [].
    """

    def sorter(c: MentionCandidate):
        types = set(kb.cui_to_entity[c.concept_id].types)

        # if the alias is all caps, short and all words, it's probably an acronym
        # sometimes these are common words, like "MIX" (C1421951), "HOT" (C1424212) and "LIGHT" (C1420817)
        if (
            c.aliases[0].upper() == c.aliases[0]
            and len(c.aliases[0]) < 10
            and c.aliases[0].isalpha()
            # a non-common-word symbol will often have both upper and lower as aliases, e.g. ['NADP', 'nadp']
            # both which will have the same similarity score (since tfidf was trained on lower())
            and len(c.aliases) == 1
        ):
            return min_similarity - 0.1

        # sort non-preferred-types to the bottom
        if not has_intersection(types, list(PREFERRED_UMLS_TYPES.keys())):
            # prefer slightly over potentially common word symbols
            return min_similarity + 0.11

        # if most preferred types, bump up its "similarity"
        if has_intersection(types, list(MOST_PREFERRED_UMLS_TYPES.keys())):
            return max([1, c.similarities[0] + 0.2])

        return c.similarities[0]

    ok_candidates = sorted(
        [
            c
            for c in candidates
            if c.similarities[0] >= min_similarity
            and c.concept_id not in cui_suppressions
            and not has_intersection(
                UMLS_NAME_SUPPRESSIONS,
                kb.cui_to_entity[c.concept_id].canonical_name.split(" "),
            )
            and len(c.aliases[0].replace(" ", "")) > 2  # avoid silly short matches
        ],
        key=sorter,
        reverse=True,
    )

    return ok_candidates[0] if len(ok_candidates) > 0 else None


def clean_umls_name(
    cui: str,
    canonical_name: str,
    aliases: Sequence[str],
    type_ids: Sequence[str],
    is_composite: bool,
    overrides: dict[str, str] = UMLS_NAME_OVERRIDES,
) -> str:
    """
    Cleans up UMLS names, potentially choosing an alias over the canonical name

    - prefer shorter names
    - prefer names that are `XYZ protein` vs `protein, XYZ`
    - prefer names that start with the same word/letter as the canonical name

    Args:
        cui (str): cui
        canonical_name (str): canonical name
        aliases (list[str]): aliases
        overrides (dict[str, str], optional): overrides. Defaults to UMLS_NAME_OVERRIDES.
    """
    if cui in overrides:
        return overrides[cui]

    name_words = canonical_name.split(" ")

    # prefer canonical name if:
    # - not composite,
    # - not a stupidly long name (e.g. https://uts.nlm.nih.gov/uts/umls/concept/C4086713),
    # - and not a gene/protein
    if (
        not is_composite
        and len(name_words) < 5
        and not has_intersection(type_ids, list(UMLS_GENE_PROTEIN_TYPES.keys()))
    ) or (len(aliases) == 0):
        return canonical_name

    def name_sorter(a: str) -> int:
        # prefer shorter aliases that start with the same word/letter as the canonical name
        # e.g. TNFSF11 for "TNFSF11 protein, human"
        # todo: use something like tfidf
        return (
            # prefer short, but not too short, names
            (len(a) if len(a) > 3 else 20)
            # prefer same first word
            + (5 if a.split(" ")[0].lower() != name_words[0].lower() else 0)
            # prefer same first letter
            + (20 if a[0].lower() != canonical_name[0].lower() else 0)
            # prefer non-comma
            + (5 if "," in a else 0)
        )

    # if 1-2 words (+non-gene/protein) or no aliases, prefer canonical name
    if (
        len(name_words) == 1
        or (len(name_words) == 2 and name_words[1].lower() not in ["gene", "protein"])
        or len(aliases) == 0
    ):
        return canonical_name

    aliases = sorted(aliases, key=name_sorter)
    return aliases[0]
