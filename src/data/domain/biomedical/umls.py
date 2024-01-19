from typing import Sequence

from constants.umls import (
    UMLS_GENE_PROTEIN_TYPES,
    UMLS_NAME_OVERRIDES,
)
from utils.list import has_intersection


BAD_NAME_CHARS = [",", ")", "("]


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
    # ... or
    # - no aliases
    # - 1 word
    # non-gene/protein 2 word name without comma/parens/etc.
    if (
        not is_composite
        and len(name_words) < 5
        and not has_intersection(type_ids, list(UMLS_GENE_PROTEIN_TYPES.keys()))
    ) or (
        len(aliases) == 0  # no aliases
        or len(name_words) == 1  # 1 word
        # if 1-2 words (+non-gene/protein) or no aliases, prefer canonical name
        or (
            len(name_words) == 2
            and name_words[1].lower()
            # e.g. "XYZ modulator", not "XYZ gene"
            not in ["gene", "protein"]
            # but not "XYZ, modulator" (??)
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
