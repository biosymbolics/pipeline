"""
NER Patterns (SpaCy)

TODO:
- O-glc-NAcase
- Epiregulin/TGFα MAB
- Pirtobrutinib (LOXO-305)
- LP(a) Inhibitor
"""

from constants.patterns import (
    MOA_INFIXES,
    MOA_SUFFIXES,
    BIOLOGIC_INFIXES,
    BIOLOGIC_SUFFIXES,
    SMALL_MOLECULE_INFIXES,
    SMALL_MOLECULE_SUFFIXES,
)

from common.utils.re import get_or_re

MOA_PATTERNS: list = [
    *[
        [
            # {"HEAD": { "IN": ["PROPN", "NOUN", "ADJ"] }, "OP": "+"},
            {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "+"},
            {
                "LEMMA": {
                    "REGEX": f"(?i){moa_suffix}",
                },
            },
        ]
        for moa_suffix in MOA_SUFFIXES
    ],
    *[
        [
            {
                "LEMMA": {
                    "REGEX": f"(?i){moa_infix}",
                },
            },
        ]
        for moa_infix in MOA_INFIXES
    ],
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

TREATMENT_LINE_RE = (
    "(?:{1-4}[ ]?|{1-4}-{1-4})(?:l|line)+?"  # 1L, 2L, 3L, 4L, 1L+, 2L+, 3L+, 4L+
)

REGULATORY_DESIGNATION_PATTERN = [
    {"LOWER": {"IN": ["fast track", "accelerated approval"]}}
]


BIO_SUFFIX_RE = get_or_re(list(BIOLOGIC_SUFFIXES.keys()))
BIO_INFIX_RE = get_or_re(list(BIOLOGIC_INFIXES.keys()))


# https://www.fda.gov/media/93218/download
BIOSIMILAR_SUFFIX = "[a-z]{4}"

# https://cdn.who.int/media/docs/default-source/international-nonproprietary-names-(inn)/bioreview-2016-final.pdf
GLYCOSYLATION_MAIN_PATTERNS = ["alfa", "α", "beta", "β", "gamma", "γ", "delta", "δ"]
GLYCOSYLATION_SUB_PATTERN = "[1-4]{1}[a-c]{1}"  # e.g. alfa-2b
GLYCOSYLATION_RE = (
    "(?:"
    + "|".join(GLYCOSYLATION_MAIN_PATTERNS)
    + ")"
    + f"(?:-{GLYCOSYLATION_SUB_PATTERN})?"
)

# ipilimumab, elotuzumab, relatlimab-rmbw (relatlimab), mavacamten, elotuzumab, luspatercept-aamt, deucravacitinib
# maraleucel)(b, pomalidomide, apixaban, paclitaxel
BIOLOGIC_REGEXES = [
    "[a-z]{2,}" + BIO_SUFFIX_RE,
    "[a-z]{2,}" + BIO_INFIX_RE + "[a-zαβγδ]{2,}",
]

# Interest expense decreased
BIOLOGICAL_PATTERNS: list[list[dict]] = [
    [
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
        {"LOWER": {"REGEX": bio_re}},
        {"LOWER": {"REGEX": GLYCOSYLATION_RE}, "OP": "?"},
        {"LOWER": {"REGEX": BIOSIMILAR_SUFFIX}, "OP": "?"},
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
    ]
    for bio_re in BIOLOGIC_REGEXES
]

CORE_SMALL_MOLECULE_REGEXES = [
    "[a-z]+" + get_or_re(list(SMALL_MOLECULE_SUFFIXES.keys())),
    "[a-z]+" + get_or_re(list(SMALL_MOLECULE_INFIXES.keys())) + "[a-zαβγδ]+",
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


BRAND_NAME_PATTERNS: list[list[dict]] = [
    [
        {
            "TEXT": {
                "REGEX": ".+[®ª^]\\b",
            },
        },
    ]
]

INTERVENTION_SPACY_PATTERNS = [
    *[{"label": "PRODUCT", "pattern": moa_re} for moa_re in MOA_PATTERNS],
    *[{"label": "PRODUCT", "pattern": id_re} for id_re in INVESTIGATIONAL_ID_PATTERNS],
    *[{"label": "PRODUCT", "pattern": bio_re} for bio_re in BIOLOGICAL_PATTERNS],
    *[{"label": "PRODUCT", "pattern": sme_re} for sme_re in SMALL_MOLECULE_PATTERNS],
    *[{"label": "PRODUCT", "pattern": brand_re} for brand_re in BRAND_NAME_PATTERNS],
]
