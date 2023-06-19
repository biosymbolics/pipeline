"""
Utility functions for the patents client
"""
from functools import reduce
import re
from typing import Optional
from datetime import date
import polars as pl

from common.utils.list import has_intersection
from common.utils.re import get_or_re, remove_extra_spaces
from common.ner import classify_by_keywords
from .constants import (
    COMPANY_NAME_SUPPRESSIONS,
    COUNTRIES,
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


ASSIGNEE_NAME_MAP = {
    "massachusetts inst technology": "Massachusetts Institute of Technology",
    "hoffmann la roche": "Roche",
    "boston scient": "Boston Scientific",
    "boston scient neuromodulation": "Boston Scientific",
    "boston scimed": "Boston Scientific",
    "lilly co eli": "Eli Lilly",
    "glaxo": "GlaxoSmithKline",
    "merck sharp & dohme": "Merck",
    "merck sharp & dohme de espana": "Merck",
    "merck frosst": "Merck",
    "california inst of techn": "CalTech",
    "cedars sinai center": "Mount Sinai",
    "sinai school medicine": "Mount Sinai",
    "icahn school med mount sinai": "Mount Sinai",
    "mount sinai hospital": "Mount Sinai",
    "ity mount sinai school of medi": "Mount Sinai",
    "sinai health sys": "Mount Sinai",
    "medtronic vascular": "Medtronic",
    "sloan kettering inst cancer": "Sloan Kettering",
    "memorial sloan kettering cancer center": "Sloan Kettering",
    "sloan kettering inst cancer res": "Sloan Kettering",
    "sanofi aventis": "Sanofi",
    "sanofis aventis": "Sanofi",
    "basf ag": "Basf",
    "basf se": "Basf",
    "3m innovative properties co": "3M",
    "abbott lab": "Abbott",
    "abbott diabetes": "Abbott",
    "medical res council technology": "Medical Research Council",
    "med res council": "Medical Research Council",
    "medical res council": "Medical Research Council",
    "medical res council tech": "Medical Research Council",
    "mayo foundation": "Mayo Clinic",
    "conopco dba unilever": "Unilever",
    "dow chemical co": "Dow Chemical",
    "dow agrosciences llc": "Dow Chemical",
    "ge": "GE",
    "lg": "LG",
    "nat cancer ct": "National Cancer Center",
    "samsung life public welfare": "Samsung",
    "verily life": "Verily",
    "lg chemical ltd": "LG",
    "isis innovation": "Isis",
    "broad": "Broad Institute",
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

    def remove_suppressions(_assignee):
        """
        Remove suppressions (generic terms like LLC, country, etc),
        Examples:
            - Matsushita Electric Ind Co Ltd -> Matsushita
            - MEDIMMUNE LLC -> Medimmune
        """
        suppress_re = (
            "\\b" + get_or_re([*COMPANY_NAME_SUPPRESSIONS, *COUNTRIES]) + "\\b"
        )
        return re.sub("(?i)" + suppress_re, "", _assignee).rstrip("&|of")

    def is_excepted(cleaned):
        # avoid reducing names to near nothing
        # e.g. "Med Inst", "Lt Mat"
        # TODO: make longer (4-5 chars) but check for common word or not
        return len(cleaned) < 3 or cleaned.lower() in ["univ"]

    cleaning_steps = [remove_suppressions, remove_extra_spaces]
    cleaned = reduce(lambda x, func: func(x), cleaning_steps, assignee)

    if is_excepted(cleaned):
        return assignee.title()

    # see if there is an explicit name mapping
    if has_intersection(
        [assignee.lower(), cleaned.lower()], list(ASSIGNEE_NAME_MAP.keys())
    ):
        return (
            ASSIGNEE_NAME_MAP.get(assignee.lower())
            or ASSIGNEE_NAME_MAP[cleaned.lower()]
        )

    return cleaned.title()


def get_patent_attributes(titles: pl.Series) -> pl.Series:
    """
    Get patent attributes from a title
    """
    return classify_by_keywords(titles, PATENT_ATTRIBUTE_MAP, None)
