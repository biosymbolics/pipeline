"""
Utility functions for the patents client
"""
from typing import Optional
from datetime import date

MAX_PATENT_LIFE = 20


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
