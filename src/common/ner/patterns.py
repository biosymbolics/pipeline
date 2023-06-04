"""
NER Patterns (SpaCy)
"""

from constants.patterns import (
    MOA_INFIXES,
    MOA_SUFFIXES,
    BIOLOGIC_INFIXES,
    BIOLOGIC_SUFFIXES,
    INDICATION_REGEXES,
    INDICATION_MODIFIER_REGEXES,
    SMALL_MOLECULE_INFIXES,
    SMALL_MOLECULE_SUFFIXES,
)

from common.utils.re import (
    get_or_re,
    WORD_DIGIT_CHAR_RE as WD_CHAR_RE,
    COPYRIGHT_SYM,
    REGISTERED_SYM,
)

# end-of-entity regex
EOE_RE = "\\b" + ".*"

# start-of-entity regex
SOE_RE = ".*"

# also ENTITY Opdualag tag: NNP pos: PROPN dep: nmod lemma: Opdualag morph: Number=Sing prob: -20.0 head: approval span: [of, ,, nivolumab, ,]
MOA_PATTERNS: list = [
    *[
        [
            {
                "POS": {"IN": ["PROPN", "NOUN", "ADJ"]},
                "OP": "*",
            },
            {
                "LOWER": {
                    "REGEX": SOE_RE + moa_suffix + EOE_RE,
                },
            },
            # UNKNOWN luspatercept-aamt tag: JJ pos: ADJ dep: dep lemma: luspatercept-aamt morph: Degree=Pos prob: -20.0 head: Reblozyl span: [(, )]
            # UNKNOWN luspatercept-aamt tag: JJ pos: ADJ dep: ROOT lemma: luspatercept-aamt morph: Degree=Pos prob: -20.0 head: luspatercept-aamt span: [Reblozyl, ,, (, ), ,, 2031, +, lenalidomide, .]
            # UNKNOWN luspatercept-aamt tag: JJ pos: ADJ dep: dep lemma: luspatercept-aamt morph: Degree=Pos prob: -20.0 head: Reblozyl span: [(, )]
            {
                "POS": {"IN": ["PROPN", "NOUN", "ADJ"]},
                "OP": "*",
            },
        ]
        for moa_suffix in MOA_SUFFIXES
    ],
    *[
        [
            {
                "POS": {"IN": ["PROPN", "NOUN", "ADJ"]},
                "OP": "*",
            },
            {
                "LOWER": {
                    "REGEX": SOE_RE + f"{moa_infix}{WD_CHAR_RE}*" + EOE_RE,
                },
            },
            {
                "POS": {"IN": ["PROPN", "NOUN", "ADJ"]},
                "OP": "*",
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

# https://www.fda.gov/media/93218/download
BIOSIMILAR_SUFFIX = "-?[a-z]{4}"

# https://cdn.who.int/media/docs/default-source/international-nonproprietary-names-(inn)/bioreview-2016-final.pdf
# e.g. alfa-2b
GLYCOSYLATION_MAIN_PATTERNS = ["alfa", "α", "beta", "β", "gamma", "γ", "delta", "δ"]
GLYCOSYLATION_SUB_PATTERN = "[1-4]{1}[a-c]{1}"
GLYCOSYLATION_RE = (
    "(?:"
    + "|".join(GLYCOSYLATION_MAIN_PATTERNS)
    + ")"
    + f"(?:-{GLYCOSYLATION_SUB_PATTERN})?"
)


# ipilimumab, elotuzumab, relatlimab-rmbw (relatlimab), mavacamten, elotuzumab, luspatercept-aamt, deucravacitinib
BIOLOGIC_REGEXES = [
    SOE_RE + WD_CHAR_RE + "{2,}" + get_or_re(list(BIOLOGIC_SUFFIXES.keys())) + EOE_RE,
    SOE_RE
    + WD_CHAR_RE
    + "{2,}"
    + get_or_re(list(BIOLOGIC_INFIXES.keys()))
    + f"{WD_CHAR_RE}{2,}"
    + EOE_RE,
]

# luspatercept-aamt
BIOLOGICAL_PATTERNS: list[list[dict]] = [
    [
        # {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
        {"LOWER": {"REGEX": bio_re}},  # , "POS": {"IN": ["PROPN", "NOUN"]}
        # {"LOWER": {"REGEX": GLYCOSYLATION_RE}, "OP": "?"},
        {"LOWER": {"REGEX": BIOSIMILAR_SUFFIX}, "OP": "?"},
        # {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
    ]
    for bio_re in BIOLOGIC_REGEXES
]

SMALL_MOLECULE_REGEXES = [
    SOE_RE
    + f"{WD_CHAR_RE}+"
    + get_or_re(list(SMALL_MOLECULE_SUFFIXES.keys()))
    + EOE_RE,
    SOE_RE
    + f"{WD_CHAR_RE}+"
    + get_or_re(list(SMALL_MOLECULE_INFIXES.keys()))
    + f"{WD_CHAR_RE}+"
    + EOE_RE,
]

SMALL_MOLECULE_PATTERNS: list[list[dict]] = [
    [
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
        {"LOWER": {"REGEX": sm_re}, "POS": {"IN": ["PROPN", "NOUN"]}},
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
    ]
    for sm_re in SMALL_MOLECULE_REGEXES
]

# Additional: infrequent (tf/idf) PROPN?
BRAND_NAME_PATTERNS: list[list[dict]] = [
    [
        {
            "TEXT": {
                "REGEX": SOE_RE
                + WD_CHAR_RE
                + "{5,}[ ]?[{COPYRIGHT_SYM}{REGISTERED_SYM}]"
                + EOE_RE
            },
        },
    ]
]


ALL_PATTERNS = [
    *MOA_PATTERNS,
    *INVESTIGATIONAL_ID_PATTERNS,
    *BIOLOGICAL_PATTERNS,
    *SMALL_MOLECULE_PATTERNS,
]

INTERVENTION_SPACY_PATTERNS = [
    *[
        {"label": "PRODUCT", "pattern": pattern}
        for pattern in INVESTIGATIONAL_ID_PATTERNS
    ],
    *[{"label": "PRODUCT", "pattern": pattern} for pattern in BIOLOGICAL_PATTERNS],
    *[{"label": "PRODUCT", "pattern": pattern} for pattern in SMALL_MOLECULE_PATTERNS],
    *[{"label": "PRODUCT", "pattern": pattern} for pattern in BRAND_NAME_PATTERNS],
    *[{"label": "PRODUCT", "pattern": pattern} for pattern in MOA_PATTERNS],
    # from en_ner_bc5cdr_md model
    {
        "label": "PRODUCT",
        "pattern": [{"ENT_TYPE": "CHEMICAL"}],
    },
]


"""
Indication patterns
"""

INDICATION_REGEXES = [
    f"(?:{WD_CHAR_RE}*\\s)*"
    + get_or_re(INDICATION_MODIFIER_REGEXES)
    + "*"
    + get_or_re(INDICATION_REGEXES)
    + "+"
    + EOE_RE,
]

INDICATION_PATTERNS: list[list[dict]] = [
    [
        {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "*"},
        {"LOWER": {"REGEX": ind_re}},
        {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "*"},
    ]
    for ind_re in INDICATION_REGEXES
]

INDICATION_SPACY_PATTERNS = [
    *[{"label": "DISEASE", "pattern": pattern} for pattern in INDICATION_PATTERNS],
]
