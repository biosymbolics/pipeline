"""
Pattern constants
"""

__all__ = [
    "INDICATION_REGEXES",
    "INDICATION_MODIFIER_REGEXES",
    "IUPAC_RE",
    "IUPAC_STRINGS",
    "MOA_INFIXES",
    "MOA_PREFIXES",
    "MOA_SUFFIXES",
]

from .indication import INDICATION_MODIFIER_REGEXES, INDICATION_REGEXES
from .moa import (
    MOA_INFIXES,
    MOA_PREFIXES,
    MOA_SUFFIXES,
)
from .iupac import IUPAC_RE, IUPAC_STRINGS
