"""
Utility functions for the patents client
"""
from typing import Optional
from datetime import date
import polars as pl

from .constants import MAX_PATENT_LIFE


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
