from functools import partial, reduce
from typing import Iterable
import re
import logging

from utils.re import get_or_re, remove_extra_spaces
from constants.patents import (
    COMPANY_MAP,
    COMPANY_SUPPRESSIONS,
    COMPANY_SUPPRESSIONS_DEFINITE,
)


EXCEPTION_TERMS = [
    "agency",
    "council",
    "gen",
    "korea",
    "life",
    "univ",
]


def clean_assignees(assignees: list[str]) -> Iterable[str]:
    """
    Clean an assignee name
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases

    Args:
        assignees (list[str]): List of assignee names
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
        suppress_re = "\\b" + get_or_re(suppressions) + "\\b"

        for term in terms:
            yield re.sub("(?i)" + suppress_re, "", term).rstrip("&[ ]*")

    def get_mapping(clean_assignee: str, og_assignee: str, key: str) -> str | None:
        """
        See if there is an explicit name mapping on cleaned or original assignee
        """
        to_check = [clean_assignee, og_assignee]
        has_mapping = any(
            [re.findall("(?i)" + "\\b" + key + "\\b", check) for check in to_check]
        )
        if has_mapping:
            return key
        return None

    def rewrite(assignees: list[str], lookup_map) -> Iterable[str]:
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

    def handle_exception(terms: list[str]) -> Iterable[str]:
        """
        Avoid reducing names to near nothing
        e.g. "Med Inst", "Lt Mat"
        TODO: make longer (4-5 chars) but check for common word or not
        """
        exceptions = [
            len(term) < 3 or term.lower() in EXCEPTION_TERMS for term in terms
        ]

        steps = [
            partial(remove_suppressions, only_definite=True),
            remove_extra_spaces,
        ]
        for term, is_exception in zip(terms, exceptions):
            _term = reduce(
                lambda x, func: (func(x) if is_exception else term), steps, term
            )
            yield _term

    def title(assignees: list[str]) -> Iterable[str]:
        for assignee in assignees:
            yield assignee.title()

    cleaning_steps = [
        remove_suppressions,
        remove_extra_spaces,
        handle_exception,
        title,
    ]
    cleaned = reduce(lambda x, func: func(x), cleaning_steps, assignees)
    lookup_map = dict(zip(cleaned, assignees))

    return rewrite(cleaned, lookup_map)
