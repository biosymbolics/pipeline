from functools import partial, reduce
from typing import Iterable
import re
import logging

from utils.re import get_or_re, remove_extra_spaces
from constants.patents import (
    COMPANY_MAP,
    COMPANY_SUPPRESSIONS,
    COMPANY_SUPPRESSIONS_DEFINITE,
    OWNER_TERM_MAP,
)


EXCEPTION_TERMS = [
    "agency",
    "council",
    "gen",
    "korea",
    "life",
    "univ",
]


def clean_owners(owners: list[str]) -> Iterable[str]:
    """
    Clean owner names
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases

    Args:
        owners (list[tuple[str, str]]): List of owner names
    """

    def remove_suppressions(terms: list[str], only_definite=False) -> Iterable[str]:
        """
        Remove suppressions (generic terms like LLC, country, etc),
        Examples:
            - Matsushita Electric Ind Co Ltd -> Matsushita
            - MEDIMMUNE LLC -> Medimmune
        """
        suppressions = (
            COMPANY_SUPPRESSIONS_DEFINITE if only_definite else COMPANY_SUPPRESSIONS
        )
        suppress_re = r"\b" + get_or_re(suppressions) + r"\b"

        for term in terms:
            yield re.sub("(?i)" + suppress_re, "", term, flags=re.DOTALL).rstrip(
                "&[ ]*"
            )

    def get_mapping(clean_assignee: str, og_assignee: str, key: str) -> str | None:
        """
        See if there is an explicit name mapping on cleaned or original assignee
        """
        to_check = [clean_assignee, og_assignee]
        has_mapping = any(
            [re.findall("(?i)" + r"\b" + key + r"\b", check) for check in to_check]
        )
        if has_mapping:
            return key
        return None

    def normalize_terms(assignees: list[str]) -> Iterable[str]:
        """
        Normalize terms (e.g. "lab" -> "laboratory", "univ" -> "uni")
        """
        term_map_set = set(OWNER_TERM_MAP.keys())

        def __normalize(cleaned: str):
            terms_to_rewrite = list(term_map_set.intersection(cleaned.split()))
            if len(terms_to_rewrite) > 0:
                _assignee = assignee
                for term in terms_to_rewrite:
                    _assignee = re.sub(
                        rf"\y{term}\y", f" {OWNER_TERM_MAP[term]} ", cleaned
                    ).strip()
                return _assignee

            return assignee

        for assignee in assignees:
            yield __normalize(assignee)

    def rewrite_by_lookup(assignees: list[str], lookup_map) -> Iterable[str]:
        def __map(cleaned: str):
            og_assignee = lookup_map[cleaned]
            mappings = [
                key
                for key in COMPANY_MAP.keys()
                if get_mapping(cleaned, og_assignee, key)
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
    lookup_map = dict(zip(cleaned, owners))

    return rewrite_by_lookup(cleaned, lookup_map)
