from typing import Sequence
from prisma.enums import BiomedicalEntityType

from constants.umls import (
    MOST_PREFERRED_UMLS_TYPES,
    UMLS_CUI_ALIAS_SUPPRESSIONS,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_GENE_PROTEIN_TYPES,
    UMLS_NAME_OVERRIDES,
    UMLS_NAME_SUPPRESSIONS,
    UMLS_NON_COMPOSITE_SUPPRESSION,
    UMLS_TO_ENTITY_TYPE,
)
from utils.list import has_intersection


# can be any single char
BAD_NAME_CHARS = [",", ")", "(", "[", "]", "-"]


def clean_umls_name(
    cui: str,
    canonical_name: str,
    aliases: Sequence[str],
    type_ids: Sequence[str],
    is_composite: bool = False,
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
        type_ids (list[str]): type ids
        is_composite (bool, optional): whether the name is composite. Defaults to False.
        overrides (dict[str, str], optional): overrides. Defaults to UMLS_NAME_OVERRIDES.
    """
    if cui in overrides:
        return overrides[cui]

    name_words = canonical_name.split(" ")

    # prefer canonical name if:
    # - not composite,
    # - not a stupidly long name (e.g. https://uts.nlm.nih.gov/uts/umls/concept/C4086713),
    # - and not a gene/protein, nor having suppressed chars
    # ... or
    # - no aliases
    # - 1 word
    # non-gene/protein 2 word name without comma/parens/etc.
    # TODO: seems kinda arbitrary
    if (
        len(aliases) == 0
        or len(name_words) == 1
        or (
            not is_composite
            and len(name_words) < 5
            and not has_intersection(type_ids, list(UMLS_GENE_PROTEIN_TYPES.keys()))
        )
        or (
            # if 2 words, non-gene/non-protein, without forbidden chars
            len(name_words) == 2
            # e.g. "XYZ modulator", not "XYZ gene"
            and name_words[1].lower() not in ["gene", "protein"]
            # but not "XYZ, modulator"
            and not has_intersection(BAD_NAME_CHARS, list(canonical_name))
        )
    ):
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
            + (5 if has_intersection(BAD_NAME_CHARS, list(a)) else 0)
        )

    aliases = sorted(aliases, key=name_sorter)
    return aliases[0]


def is_umls_suppressed(
    id: str, canonical_name: str, matching_aliases: Sequence[str], is_composite: bool
) -> bool:
    """
    Whether a UMLS concept should be suppressed from candidate linking / ancestors /etc

    Args:
        id (str): cui
        canonical_name (str): canonical name
    """
    if id in UMLS_CUI_SUPPRESSIONS:
        return True

    if has_intersection(canonical_name.split(" "), UMLS_NAME_SUPPRESSIONS):
        return True

    # suppress if matching alias is suppressed for that cui
    # which is useful to avoid, for example, "ice" matching methamphetamine
    # (see https://uts.nlm.nih.gov/uts/umls/concept/C0025611)
    lower_matching_aliases = [a.lower() for a in matching_aliases]
    if (
        len(lower_matching_aliases) > 0
        and id in UMLS_CUI_ALIAS_SUPPRESSIONS
        and has_intersection(lower_matching_aliases, UMLS_CUI_ALIAS_SUPPRESSIONS[id])
    ):
        return True

    if not is_composite and id in UMLS_NON_COMPOSITE_SUPPRESSION:
        return True

    return False


def tuis_to_entity_type(tuis: Sequence[str]) -> BiomedicalEntityType:
    """
    Given a list of tuis, return the corresponding entity type

    Args:
        tuis (list[str]): list of tuis
    """
    # else, grab known tuis
    known_tuis = [tui for tui in tuis if tui in UMLS_TO_ENTITY_TYPE]

    if len(known_tuis) == 0:
        return BiomedicalEntityType.UNKNOWN

    # chose preferred tuis
    preferred_tuis = [tui for tui in known_tuis if tui in MOST_PREFERRED_UMLS_TYPES]

    # if no preferred types, return first known tui
    if len(preferred_tuis) == 0:
        return UMLS_TO_ENTITY_TYPE[known_tuis[0]]

    return UMLS_TO_ENTITY_TYPE[preferred_tuis[0]]
