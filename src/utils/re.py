"""
Regex utilities
"""

import regex as re
from typing import Iterable, Literal, Optional, Sequence, Union
import logging

from utils.list import is_sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RE_STANDARD_FLAGS = re.IGNORECASE | re.MULTILINE


# (includes greek chars)
def WORD_CHAR_RE(additional_chars: list[str] = []):
    """
    Returns a regex for a single word char

    Args:
        additional_chars (list[str]): additional chars to include (as string literals)
    """
    return "[\\w\u0370-\u03FF" + "".join(additional_chars) + "]"


WORD_DIGIT_RE = rf"(?:{WORD_CHAR_RE()}|\d)"


COPYRIGHT_SYM = "\u00A9"  # ©
REGISTERED_SYM = "\u00AE"  # ®
TM_SYM = "\u2122"  # ™

LEGAL_SYMBOLS = [COPYRIGHT_SYM, REGISTERED_SYM, TM_SYM]

HACKY_STEMMING_RE = (
    ".?(?:ing|e|ied|ed|er|or|en|ion|ist|ly|able|ive|al|ic|ous|y|ate|at|ry|y|ie)*s?"
)

ReCount = Union[Literal["*", "+", "?"], int]


def get_hacky_stem_re(terms: list[str], **kwargs) -> str:
    """
    Get re that matches terms with hacky stemming
    (requires that terms be stemmed; this adds the "s", "ing", "ed", etc suffixes)
    """
    return "(?i)" + f"^{get_or_re(terms, **kwargs)}{HACKY_STEMMING_RE}$"


def get_or_re(
    _re_strs: Sequence[str] | set[str],
    count: Optional[ReCount] = None,
    upper: Optional[int] = None,
    permit_trailing_space: bool = False,
    enforce_word_boundaries: bool = False,
    permit_plural: bool = True,
    word_boundary_char: str = "\\b",  # \y for postgres
) -> str:
    """
    Gets a regex that ORs a list of regexes

    Args:
        re_strs (list[str]): list of regexes
        count (Optional[ReCount]): count to apply to regex (defaults to None, which is effectively {1})
        upper (Optional[int]): upper bound for count (defaults to None, which is effectively {1, 100})
        permit_trailing_space (bool): whether to permit trailing space between each re (defaults to False)
        enforce_word_boundaries (bool): whether to enforce word boundaries (defaults to False)
    """
    if not is_sequence(_re_strs):
        raise Exception("re_strs must be a sequence")

    re_strs = [*_re_strs]
    if permit_plural:
        re_strs = [re_str + r"s?" for re_str in re_strs]

    wbc = word_boundary_char
    if enforce_word_boundaries:
        re_strs = [f"{wbc}{re_str}{wbc}" for re_str in re_strs]

    if permit_trailing_space:
        re_strs = [re_str + r"\s*" for re_str in re_strs]

    base_re = f"(?:{'|'.join(re_strs)})"

    if count is None:
        return base_re
    if isinstance(count, int):
        return base_re + f"{{{str(count)},{str(upper or '100')}}}"

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


def remove_extra_spaces(terms: list[str] | Iterable[str]) -> Iterable[str]:
    """
    Removes extra spaces from terms
    (also strips)

    Args:
        terms: list of terms from which to remove extra spaces
    """
    if isinstance(terms, str):
        raise Exception("terms must be a list or iterable")

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


def expand_re(re_str: str, max_len: int = 1000) -> list[str]:
    """
    Expands a regex into a list of strings

    Args:
        re_str (str): regex to expand
        max_len (int): maximum number of strings to generate for each regex, above which the method will throw an exception

    Example:
        expand_re("fab(?: region)?") -> ['fab', 'fab region']
    """
    # lazy import
    import exrex

    try:
        if exrex.count(re_str) < max_len:
            return list(exrex.generate(re_str))

        else:
            raise Exception("Regex too complex to expand")
    except Exception as e:
        logger.error("Failed to expand regex %s: %s. Returning raw str.", re_str, e)
        return [re_str]


def expand_res(re_strs: list[str]) -> list[list[str]]:
    """
    Expands a list of regexes into a list of lists of strings

    Args:
        re_strs (list[str]): list of regexes
    """
    return [expand_re(re_str) for re_str in re_strs]
