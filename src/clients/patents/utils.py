"""
Utility functions for the patents client
"""
from typing import Optional
from datetime import date, timedelta
import polars as pl

from .constants import MAX_PATENT_LIFE, STALE_YEARS


def get_max_priority_date(min_patent_years: Optional[int] = 0) -> date:
    """
    Outputs max priority date as number YYYYMMDD (e.g. 20211031)

    Args:
        min_patent_years (Optional[int], optional): min number of years remaining on patent. Defaults to 0.
    """
    years = MAX_PATENT_LIFE - (min_patent_years or 0)
    max_priority_date = date.today() - timedelta(days=years * 365)

    return max_priority_date


def get_patent_years() -> pl.Expr:
    """
    Get the number of years remaining on a patent for a Series of dates
    """
    current_year = date.today().year
    expr = (
        pl.lit(MAX_PATENT_LIFE)
        - (pl.lit(current_year) - pl.col("priority_date").dt.year())
    ).clip(lower_bound=0, upper_bound=MAX_PATENT_LIFE)
    return expr


def is_patent_stale() -> pl.Expr:
    """
    Check if patent is stale (i.e. possibly abandoned)
    """
    expr = (pl.lit(date.today()) - pl.col("last_trial_update")) > pl.duration(
        days=365 * STALE_YEARS
    )
    return expr
