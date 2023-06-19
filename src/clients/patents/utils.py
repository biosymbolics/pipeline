"""
Utility functions for the patents client
"""
from functools import reduce
import logging
import re
from typing import Optional
from datetime import date
import polars as pl

from common.utils.re import get_or_re, remove_extra_spaces
from common.ner import classify_by_keywords
from .constants import (
    COMPANY_SUPPRESSIONS,
    COMPANY_SUPPRESSIONS_DEFINITE,
    PATENT_ATTRIBUTE_MAP,
    MAX_PATENT_LIFE,
)


def get_max_priority_date(min_patent_years: Optional[int] = 0) -> int:
    """
    Outputs max priority date as number YYYYMMDD (e.g. 20211031)

    Args:
        min_patent_years (Optional[int], optional): min number of years remaining on patent. Defaults to 0.
    """
    # e.g. 2021 - 20 = 2001
    priority_year = date.today().year - MAX_PATENT_LIFE

    # e.g. 2001 + min of 10 yrs remaining = 2011
    max_priority_year = priority_year + (min_patent_years or 0)

    # e.g. 2001 -> 20010000
    as_number = max_priority_year * 10000
    return as_number


def get_patent_years(priority_dates_column: str) -> pl.Expr:
    """
    Get the number of years remaining on a patent for a Series of dates

    Args:
        priority_dates_column (str): Column name of priority dates of the patents
    """
    current_year = date.today().year
    expr = (
        pl.lit(MAX_PATENT_LIFE)
        - (pl.lit(current_year) - pl.col(priority_dates_column).dt.year())
    ).clip(lower_bound=0, upper_bound=MAX_PATENT_LIFE)
    return expr


ASSIGNEE_MAP = {
    "massachusetts inst technology": "Massachusetts Institute of Technology",
    "roche": "Roche",
    "boston scient": "Boston Scientific",
    "boston scimed": "Boston Scientific",
    "lilly co eli": "Eli Lilly",
    "glaxo": "GlaxoSmithKline",
    "merck sharp & dohme": "Merck",
    "merck frosst": "Merck",
    "california inst of techn": "CalTech",
    "sinai": "Mount Sinai",
    "medtronic": "Medtronic",
    "sloan kettering": "Sloan Kettering",
    "sanofi": "Sanofi",
    "sanofis": "Sanofi",
    "basf": "Basf",
    "3m": "3M",
    "abbott": "Abbott",
    "medical res council": "Medical Research Council",
    "mayo": "Mayo Clinic",  # FPs
    "unilever": "Unilever",
    "gen eletric": "GE",
    "ge": "GE",
    "lg": "LG",
    "nat cancer ct": "National Cancer Center",
    "samsung": "Samsung",
    "verily": "Verily",
    "isis": "Isis",
    "broad": "Broad Institute",
    "childrens medical center": "Childrens Medical Center",
    "us gov": "US Government",
    "koninkl philips": "Philips",
    "koninklijke philips": "Philips",
    "max planck": "Max Planck",
    "novartis": "Novartis",
}


def clean_assignee(assignee: str) -> str:
    """
    Clean an assignee name
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases

    Args:
        assignee (str): assignee name
    """

    def remove_suppressions(string, only_definite=False):
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
        return (
            re.sub("(?i)" + suppress_re, "", string).rstrip("&[ ]*").strip("o|Of[ ]*")
        )

    def get_mapping(string, key):
        """
        See if there is an explicit name mapping on cleaned or original assignee
        """
        to_check = [assignee, string]
        has_mapping = any(
            [re.findall("(?i)" + "\\b" + key + "\\b", check) for check in to_check]
        )
        if has_mapping:
            return key
        return None

    def handle_mapping(string):
        mappings = [key for key in ASSIGNEE_MAP.keys() if get_mapping(string, key)]
        if len(mappings) > 0:
            logging.info("Found mapping for assignee: %s -> %s", string, mappings[0])
            return ASSIGNEE_MAP[mappings[0]]
        return string

    def handle_exception(string):
        """
        Avoid reducing names to near nothing
        e.g. "Med Inst", "Lt Mat"
        TODO: make longer (4-5 chars) but check for common word or not
        """
        is_exception = len(string) < 3 or string.lower() in [
            "agency",
            "council",
            "gen",
            "korea",
            "life",
            "univ",
        ]
        return (
            remove_extra_spaces(remove_suppressions(assignee, True))
            if is_exception
            else string
        )

    cleaning_steps = [
        remove_suppressions,
        remove_extra_spaces,
        handle_exception,
        handle_mapping,
    ]
    cleaned = reduce(lambda x, func: func(x), cleaning_steps, assignee)

    return cleaned.title()


def get_patent_attributes(titles: pl.Series) -> pl.Series:
    """
    Get patent attributes from a title
    """
    return classify_by_keywords(titles, PATENT_ATTRIBUTE_MAP, None)
