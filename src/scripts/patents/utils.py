from functools import reduce
from typing import Iterable
import regex as re
import logging

from constants.patents import (
    COMPANY_MAP,
    OWNER_SUPPRESSIONS,
    OWNER_TERM_MAP,
)
from core.ner.utils import cluster_terms
from utils.re import get_or_re, remove_extra_spaces, RE_STANDARD_FLAGS

RE_FLAGS = RE_STANDARD_FLAGS


def clean_owners(owners: list[str]) -> list[str]:
    """
    Clean owner names
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases
    - normalizes terms (e.g. "lab" -> "laboratory", "univ" -> "university")
    - applies overrides (e.g. "biogen ma" -> "biogen")
    - then does clustering


    Args:
        owners (list[tuple[str, str]]): List of owner names
    """

    def remove_suppressions(terms: list[str]) -> Iterable[str]:
        """
        Remove suppressions (generic terms like LLC, country, etc),
        Examples:
            - Matsushita Electric Ind Co Ltd -> Matsushita
            - MEDIMMUNE LLC -> Medimmune
            - University Of Alabama At Birmingham  -> University Of Alabama
            - University of Colorado, Denver -> University of Colorado
        """
        post_suppress_re = rf"(?:(?:\s+|,){get_or_re(OWNER_SUPPRESSIONS)}\b)"
        pre_suppress_re = "^the"
        suppress_re = rf"(?:{pre_suppress_re}|{post_suppress_re}|((?:,|at) .*$)|\(.+\))"

        for term in terms:
            yield re.sub(suppress_re, "", term, flags=RE_FLAGS).rstrip("&[ .,]*")

    def normalize_terms(assignees: list[str]) -> Iterable[str]:
        """
        Normalize terms (e.g. "lab" -> "laboratory", "univ" -> "university")
        """
        term_map_set = set(OWNER_TERM_MAP.keys())

        def __normalize(cleaned: str):
            terms_to_rewrite = list(term_map_set.intersection(cleaned.lower().split()))
            if len(terms_to_rewrite) > 0:
                _assignee = assignee
                for term in terms_to_rewrite:
                    _assignee = re.sub(
                        rf"\b{term}\b",
                        f" {OWNER_TERM_MAP[term.lower()]} ",
                        cleaned,
                        flags=RE_FLAGS,
                    ).strip()
                return _assignee

            return assignee

        for assignee in assignees:
            yield __normalize(assignee)

    def get_lookup_mapping(
        clean_assignee: str, og_assignee: str, key: str
    ) -> str | None:
        """
        See if there is an explicit name mapping on cleaned or original assignee
        """
        to_check = [clean_assignee, og_assignee]
        has_mapping = any(
            [re.findall(rf"\b{key}\b", check, flags=RE_FLAGS) for check in to_check]
        )
        if has_mapping:
            return key
        return None

    def apply_overrides(
        assignees: list[str], cleaned_orig_map: dict[str, str]
    ) -> Iterable[str]:
        def __map(cleaned: str):
            og_assignee = cleaned_orig_map[cleaned]
            mappings = [
                key
                for key in COMPANY_MAP.keys()
                if get_lookup_mapping(cleaned, og_assignee, key)
            ]
            if len(mappings) > 0:
                logging.debug(
                    "Found mapping for assignee: %s -> %s", assignee, mappings[0]
                )
                return COMPANY_MAP[mappings[0]]
            return assignee

        for assignee in assignees:
            yield __map(assignee)

    def title(assignees: list[str]) -> Iterable[str]:
        for assignee in assignees:
            yield assignee.title()

    cleaning_steps = [
        remove_suppressions,
        normalize_terms,  # order matters
        remove_extra_spaces,
        title,
    ]
    cleaned = list(reduce(lambda x, func: func(x), cleaning_steps, owners))
    cleaned_orig_map = dict(zip(cleaned, owners))

    with_overiddes = list(apply_overrides(cleaned, cleaned_orig_map))

    norm_map: dict = cluster_terms(with_overiddes, return_dict=True)  # type: ignore
    final = [norm_map.get(owner) or owner for owner in with_overiddes]

    if len(final) != len(owners):
        raise ValueError(
            f"Length of cleaned assignees ({len(final)}) does not match length of original assignees ({len(owners)})"
        )

    return final


# alter table applications alter column priority_date type date USING priority_date::date;
BQ_TYPE_OVERRIDES = {
    "character_offset_start": "INTEGER",
    "character_offset_end": "INTEGER",
    "priority_date": "DATE",
}


def determine_bq_data_type(value, field: str | None = None):
    if field and field in BQ_TYPE_OVERRIDES:
        return BQ_TYPE_OVERRIDES[field]

    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, list):
        if len(value) > 0:
            dt = determine_bq_data_type(value[0])
            return f"{dt}[]"
        return "TEXT[]"
    else:  # default to TEXT for strings or other data types
        return "TEXT"
