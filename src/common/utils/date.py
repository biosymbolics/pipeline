"""
Utils related to dates
"""

from datetime import date, datetime


DEFAULT_FORMATTER = "%Y-%m-%d"


def parse_date(date_str: str, formatter: str = DEFAULT_FORMATTER) -> date:
    """
    Turns date str (YYYY-MM-DD, e.g. 2003-01-01) into date object

    Args:
        date_str (str): date str
        formatter (str, optional): date formatter. Defaults to DEFAULT_FORMATTER.
    """
    date_obj = datetime.strptime(date_str, formatter)
    return date_obj


def format_date(date_obj: date, formatter: str = DEFAULT_FORMATTER) -> str:
    """
    Turns date object into date str (YYYY-MM-DD, e.g. 2003-01-01)

    Args:
        date_obj (date): date object
        formatter (str, optional): date formatter. Defaults to DEFAULT_FORMATTER.
    """
    str_date = date_obj.strftime(formatter)
    return str_date


def format_date_range(start_date: date, end_date: date) -> str:
    """
    Turns start and end date into range
    e.g. "2005-01-01 TO 2010-01-01"

    Args:
        start_date (date): start date
        end_date (date): end date
    """
    range_str = f"{format_date(start_date)} TO {format_date(end_date)}"
    return range_str
