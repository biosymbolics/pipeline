"""
Patterns related to generic mechanisms of action
"""
from constants.patterns.intervention import (
    ALL_BIOLOGIC_BASE_TERMS_RE,
    INTERVENTION_PREFIXES,
    ALL_MECHANISM_BASE_TERMS_RE,
)
from utils.re import get_or_re


MOA_INFIXES = [
    *INTERVENTION_PREFIXES,
    ".+-targeted",
    ".+-binding",
]

MOA_SUFFIXES = [
    ALL_BIOLOGIC_BASE_TERMS_RE,
    ALL_MECHANISM_BASE_TERMS_RE,
    f"{get_or_re(INTERVENTION_PREFIXES, '+')}[ ]?{ALL_MECHANISM_BASE_TERMS_RE}",
]


MOA_PREFIXES = [f"{ALL_MECHANISM_BASE_TERMS_RE} of"]
