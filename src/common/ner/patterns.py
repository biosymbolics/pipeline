"""
NER Patterns (SpaCy)

TODO:
- O-glc-NAcase
- Epiregulin/TGFα MAB
- Pirtobrutinib (LOXO-305)
- LP(a) Inhibitor
"""

from constants.patterns import (
    ACTIONS,
    BIOLOGIC_INFIXES,
    BIOLOGIC_SUFFIXES,
    SMALL_MOLECULE_INFIXES,
    SMALL_MOLECULE_SUFFIXES,
)

MOA_PATTERNS: list[list[dict]] = [
    [
        # {"HEAD": { "IN": ["PROPN", "NOUN", "ADJ"] }, "OP": "+"},
        {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "+"},
        {
            "LEMMA": {
                "REGEX": f"(?i){action}",
            },
        },
    ]
    for action in ACTIONS
]

# IN supported with REGEX supported in 3.5+
INVESTIGATIONAL_ID_PATTERNS: list[list[dict]] = [
    [
        {
            "TEXT": {"REGEX": "[A-Z]{2,}[- ]?[0-9]{3,}"},  # XYZ 123, XYZ-123"
        }
    ],
    [
        {
            "TEXT": {"REGEX": "[a-zA-Z]{2,}[-]?[0-9]{3,}"},  # XYZ123, XYZ-123, Xyz-123
        }
    ],
    [
        {
            "TEXT": {
                "REGEX": "[A-Z0-9]{2,8}-[A-Z0-9]{2,8}-[A-Z0-9]{2,8}"
            },  # e.g. CAB-AXL-ADC; will miss lower-cased terms
        }
    ],
]

TRIAL_PHASE_PATTERNS = [
    {"LOWER": "preclinial"},
    {
        "LOWER": {"REGEX": "phase (?:[0-4]|i{1,3}|iv)+"}
    },  # phase 1, phase i, phase Ise i, phase I
]

REGULATORY_DESIGNATION_PATTERN = [
    {"LOWER": {"IN": ["fast track", "accelerated approval"]}}
]


def __get_or_re(re_strs: list[str]) -> str:
    return "(?:" + "|".join(re_strs) + ")"


BIO_SUFFIX_RE = __get_or_re(list(BIOLOGIC_SUFFIXES.keys()))
BIO_INFIX_RE = __get_or_re(list(BIOLOGIC_INFIXES.keys()))

CORE_BIOLOGIC_REGEXES = [
    "[a-z]{2,}" + BIO_SUFFIX_RE + "$",
    "[a-z]{2,}" + BIO_INFIX_RE + "[a-z]{2,}" + "$",
]

# https://www.fda.gov/media/93218/download
BIOSIMILAR_SUFFIX = "{[a-z]{4}}"

# https://cdn.who.int/media/docs/default-source/international-nonproprietary-names-(inn)/bioreview-2016-final.pdf
GLYCOSYLATION_MAIN_PATTERNS = ["alfa", "α", "beta", "β", "gamma", "γ", "delta", "δ"]
GLYCOSYLATION_SUB_PATTERN = "[1-4]{1}[a-c]{1}"  # e.g. alfa-2b
GLYCOSYLATION_RE = (
    "(?:"
    + "|".join(GLYCOSYLATION_MAIN_PATTERNS)
    + ")"
    + f"(?:-{GLYCOSYLATION_SUB_PATTERN})?"
)

BIOLOGIC_REGEXES = [
    *CORE_BIOLOGIC_REGEXES,
    # *[f"{bio_re} {GLYCOSYLATION_RE}" for bio_re in CORE_BIOLOGIC_REGEXES],
    # *[f"{bio_re}-{BIOSIMILAR_SUFFIX}" for bio_re in CORE_BIOLOGIC_REGEXES],
]

BIOLOGICAL_PATTERNS: list[list[dict]] = [
    [{"LOWER": {"REGEX": bio_re}}] for bio_re in BIOLOGIC_REGEXES
]

CORE_SMALL_MOLECULE_REGEXES = [
    "[a-z]+" + __get_or_re(list(SMALL_MOLECULE_SUFFIXES.keys())) + "$",
    "[a-z]+" + __get_or_re(list(SMALL_MOLECULE_INFIXES.keys())) + "[a-z]+$",
]

SMALL_MOLECULE_REGEXES = [
    *CORE_SMALL_MOLECULE_REGEXES,
    *[
        f"{sm_re}-{suffix}"
        for sm_re in CORE_SMALL_MOLECULE_REGEXES
        for suffix in SMALL_MOLECULE_SUFFIXES.keys()
    ],
]

SMALL_MOLECULE_PATTERNS: list[list[dict]] = [
    [{"LOWER": {"REGEX": sm_re}}] for sm_re in SMALL_MOLECULE_REGEXES
]
