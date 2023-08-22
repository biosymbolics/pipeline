"""
Utils related to dates
"""

from datetime import date, datetime
import logging
from typing import Any, TypeVar, cast

DEFAULT_FORMATTER = "%Y-%m-%d"

T = TypeVar("T", bound=dict[str, Any])


def date_deserialier(_object: T) -> T:
    """
    Deserializes date strings into date objects
    """
    object = _object.copy()
    for key, value in object.items():
        if "date" not in key.lower() or not isinstance(value, str):
            continue
        try:
            object[key] = parse_date(value)
        except ValueError:
            pass  # Not a date string, skip
    return cast(T, object)


def parse_date(date_str: str, formatter: str = DEFAULT_FORMATTER) -> date:
    """
    Turns date str (YYYY-MM-DD, e.g. 2003-01-01) into date object
    Will fall back to isoformat if formatter fails

    Args:
        date_str (str): date str
        formatter (str, optional): date formatter. Defaults to DEFAULT_FORMATTER.
    """
    try:
        date_obj = datetime.strptime(date_str, formatter)
    except ValueError:
        try:
            return datetime.fromisoformat(date_str)
        except:
            logging.warn("Could not parse date %s", date_str)
            raise ValueError(f"Could not parse date {date_str}")
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
