"""
Regex utilities
"""

import re
from typing import Literal, Optional, Union


def get_or_re(
    re_strs: list[str], count: Optional[Literal["+", "*", "?"]] = None
) -> str:
    """
    Gets a regex that ORs a list of regexes

    Args:
        re_strs (list[str]): list of regexes
        count (Optional[Literal["+", "*", "?"]]): count to apply to regex
    """
    return "(?:" + "|".join(re_strs) + ")" + (count or "")


WORD_CHAR_RE = "[\\w\u0370-\u03FF]"  # (includes greek chars)
WORD_DIGIT_CHAR_RE = "[\\d\\w\u0370-\u03FF]"

COPYRIGHT_SYM = "\u00A9"  # ©
REGISTERED_SYM = "\u00AE"  # ®
TM_SYM = "\u2122"  # ™

LEGAL_SYMBOLS = [COPYRIGHT_SYM, REGISTERED_SYM, TM_SYM]


ReCount = Union[Literal["*", "+", "?"], int]


def ALPHA_CHARS(count: ReCount, upper: Optional[int] = None) -> str:
    """
    Returns a regex for a sequence of alpha chars

    Args:
        count (Union[Literal["*", "+", "?"], int]): number of alpha chars
        upper (Optional[int]): upper bound for number of alpha chars (default: 1000)
    """
    if isinstance(count, int):
        return WORD_DIGIT_CHAR_RE + "{" + str(count) + "," + str(upper or "1000") + "}"
    elif upper is not None:
        raise Exception("Cannot specify upper bound unless count is int")
    return WORD_DIGIT_CHAR_RE + count


def wrap(core_re: str) -> str:
    """
    Returns a regex wrapped in a non-matching group
    """
    return "(?:" + core_re + ")"


def remove_extra_spaces(string: str) -> str:
    """
    Removes extra spaces from a string
    (also strips)

    Args:
        string (str): string to remove extra spaces from
    """
    return re.sub(r"\s+", " ", string).strip()
