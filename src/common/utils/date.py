"""
Utils related to dates
"""

from datetime import date


def format_date(_date: date, formatter: str = "%Y-%m-%d") -> str:
    """
    Turns date object into date str (YYYY-MM-DD, e.g. 2003-01-01)
    """
    str_date = _date.strftime(formatter)
    return str_date


def format_date_range(start_date: date, end_date: date) -> str:
    """
    Turns start and end date into range
    e.g. "2005-01-01 TO 2010-01-01"
    """
    range_str = f"{format_date(start_date)} TO {format_date(end_date)}"
    return range_str
