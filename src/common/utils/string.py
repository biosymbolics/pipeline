"""
String utilities
"""

from typing import Optional
import re
from functools import reduce


def get_id(string: str) -> str:
    """
    Returns the id of a string

    Args:
        string (str): string to get id of
    """
    return string.replace(" ", "_").lower()


def remove_unmatched_brackets(
    text: str, brackets: Optional[dict[str, str]] = None
) -> str:
    """
    Removes unmatched brackets from text

    Args:
        text (str): text to remove unmatched brackets from
        brackets (dict[str, str]): bracket pairs to check for

    Returns: string with unmatched brackets removed
    """
    if brackets is None:
        brackets = {"(": ")", "{": "}", "[": "]", "<": ">"}

    def replace_unmatched(text: str, bracket_pair: tuple[str, str]) -> str:
        open_sym, close_sym = bracket_pair
        # unmatched opening symbols
        text = re.sub(
            f"{re.escape(open_sym)}[^{re.escape(close_sym)} ]*[{re.escape(close_sym)} ]",
            "",
            text,
        )
        # unmatched closing symbols
        text = re.sub(
            f"[ {re.escape(open_sym)}][^{re.escape(open_sym)} ]*{re.escape(close_sym)}",
            "",
            text,
        )
        return text

    return reduce(replace_unmatched, brackets.items(), text)
