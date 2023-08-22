"""
Regex utilities
"""

import re
from typing import Iterable, Literal, Optional, Union


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
        count (Optional[ReCount]): count to apply to regex (defaults to None, which is effectively {1})
    """
    base_re = f"(?:{'|'.join(re_strs)})"

    if count is None:
        return base_re
    if isinstance(count, int):
        return base_re + f"{{{str(count)}, {str(upper or '100')}}}"

    return base_re + count


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
        return f"{WORD_CHAR_RE(additional_chars)}{{{count},{upper or '100'}}}"
    elif upper is not None:
        raise Exception("Cannot specify upper bound unless count is int")
    return WORD_CHAR_RE(additional_chars) + count


def wrap(core_re: str) -> str:
    """
    Returns a regex wrapped in a non-matching group
    """
    return "(?:" + core_re + ")"


def remove_extra_spaces(terms: list[str]) -> Iterable[str]:
    """
    Removes extra spaces from terms
    (also strips)

    Args:
        terms: list of terms from which to remove extra spaces
    """
    extra_space_patterns = {
        r"\s{2,}": " ",
        r"\s{1,},": ",",  # e.g. to address "OPSUMIT , other product"
        r"' s(\b)": r"'s\1",  # alzheimer' s disease -> alzheimer's disease
    }

    def __remove(term: str):
        for pattern, replacement in extra_space_patterns.items():
            term = re.sub(pattern, replacement, term, flags=re.DOTALL)
        return term.strip()

    for term in terms:
        yield __remove(term)
