"""
Regex utilities
"""

import re
from typing import Literal, Optional, Union


def WORD_CHAR_RE(additional_chars: list[str] = []):
    return (
        "[\\w\u0370-\u03FF" + "".join(additional_chars) + "]"
    )  # (includes greek chars)


COPYRIGHT_SYM = "\u00A9"  # ©
REGISTERED_SYM = "\u00AE"  # ®
TM_SYM = "\u2122"  # ™

LEGAL_SYMBOLS = [COPYRIGHT_SYM, REGISTERED_SYM, TM_SYM]


ReCount = Union[Literal["*", "+", "?"], int]


def get_or_re(
    re_strs: list[str], count: Optional[ReCount] = None, upper: Optional[int] = None
) -> str:
    """
    Gets a regex that ORs a list of regexes

    Args:
        re_strs (list[str]): list of regexes
        count (Optional[ReCount]): count to apply to regex
    """
    return (
        "(?:"
        + "|".join(re_strs)
        + ")"
        + "{"
        + str(count)
        + ","
        + str(upper or "100")
        + "}"
    )


def ALPHA_CHARS(
    count: ReCount, upper: Optional[int] = None, additional_chars: list[str] = []
) -> str:
    """
    Returns a regex for a sequence of alpha chars

    Args:
        count (Union[Literal["*", "+", "?"], int]): number of alpha chars
        upper (Optional[int]): upper bound for number of alpha chars (default: 100)
    """
    if isinstance(count, int):
        return (
            WORD_CHAR_RE(additional_chars)
            + "{"
            + str(count)
            + ","
            + str(upper or "100")
            + "}"
        )
    elif upper is not None:
        raise Exception("Cannot specify upper bound unless count is int")
    return WORD_CHAR_RE(additional_chars) + count


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
