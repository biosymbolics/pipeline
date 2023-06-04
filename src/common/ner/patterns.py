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
    wrap,
    ALPHA_CHARS,
    COPYRIGHT_SYM,
    REGISTERED_SYM,
)
from .utils import get_entity_re, get_infix_entity_re, get_suffix_entitiy_re, EOE_RE

# also ENTITY Opdualag tag: NNP pos: PROPN dep: nmod lemma: Opdualag morph: Number=Sing prob: -20.0 head: approval span: [of, ,, nivolumab, ,]
MOA_PATTERNS: list = [
    *[
        [
            {
                "POS": {"IN": ["PROPN", "NOUN", "ADJ"]},
                "OP": "*",
            },
            {
                "LEMMA": {
                    "REGEX": get_entity_re(moa_suffix, is_case_insensitive=True),
                },
            },
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
                    "REGEX": get_entity_re(moa_infix + ALPHA_CHARS("*")),
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
BIOSIMILAR_SUFFIX = "(?:-?[a-z]{4})"

# https://cdn.who.int/media/docs/default-source/international-nonproprietary-names-(inn)/bioreview-2016-final.pdf
# e.g. alfa-2b
GLYCOSYLATION_MAIN_PATTERNS = ["alfa", "α", "beta", "β", "gamma", "γ", "delta", "δ"]
GLYCOSYLATION_SUB_PATTERN = "[1-4]{1}[a-c]{1}"
GLYCOSYLATION_RE = wrap(
    "\\s?"
    + get_or_re(GLYCOSYLATION_MAIN_PATTERNS)
    + f"(?:[-\\s]{GLYCOSYLATION_SUB_PATTERN})?"
)

# ipilimumab, elotuzumab, relatlimab-rmbw (relatlimab), mavacamten, elotuzumab, luspatercept-aamt, deucravacitinib
BIO_SUFFIX = BIOSIMILAR_SUFFIX + "?" + GLYCOSYLATION_RE + "?" + EOE_RE
BIOLOGIC_REGEXES = [
    get_suffix_entitiy_re(
        list(BIOLOGIC_SUFFIXES.keys()), eoe_re=BIO_SUFFIX, prefix_count=2
    ),
    get_infix_entity_re(list(BIOLOGIC_INFIXES.keys())),
]

BIOLOGICAL_PATTERNS: list[list[dict]] = [
    [
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
        {"LOWER": {"REGEX": bio_re}},  # , "POS": {"IN": ["PROPN", "NOUN"]}
        {"LOWER": {"REGEX": GLYCOSYLATION_RE}, "OP": "?"},
        {"POS": {"IN": ["PROPN", "NOUN"]}, "OP": "*"},
    ]
    for bio_re in BIOLOGIC_REGEXES
]

SMALL_MOLECULE_REGEXES = [
    get_suffix_entitiy_re(list(SMALL_MOLECULE_SUFFIXES.keys()), prefix_count="+"),
    get_infix_entity_re(list(SMALL_MOLECULE_INFIXES.keys()), count="+"),
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
CR_OR_REG_SYM = f"[ ]?[{COPYRIGHT_SYM}{REGISTERED_SYM}©®]"
BRAND_NAME_RE = get_entity_re(ALPHA_CHARS(5) + CR_OR_REG_SYM, eoe_re=".*")
BRAND_NAME_PATTERNS: list[list[dict]] = [
    [
        {
            "TEXT": {
                "REGEX": BRAND_NAME_RE
            }  # e.g. "Blenrep® (belantamab mafodotin-blmf)"
        },
    ],
    [
        {
            "POS": {"IN": ["PROPN", "NOUN"]},
        },
        {
            "TEXT": {
                "REGEX": CR_OR_REG_SYM + ".*"
            },  # in "XYZ ® blah", the space after the mark is not recognized as \b
        },  # e.g. "Blenrep ®" as two different entities
    ],
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
    get_entity_re(
        get_or_re(INDICATION_MODIFIER_REGEXES, "*")
        + get_or_re(INDICATION_REGEXES, "+"),
        soe_re=f"(?:{ALPHA_CHARS('*')}\\s)*",
        is_case_insensitive=True,
    ),
]

INDICATION_PATTERNS: list[list[dict]] = [
    [
        {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "*"},
        {"LEMMA": {"REGEX": ind_re}},
        {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "*"},
    ]
    for ind_re in INDICATION_REGEXES
]

INDICATION_SPACY_PATTERNS = [
    *[{"label": "DISEASE", "pattern": pattern} for pattern in INDICATION_PATTERNS],
]


"""
All patterns
"""

ALL_PATTERNS = [
    *MOA_PATTERNS,
    *INVESTIGATIONAL_ID_PATTERNS,
    *BIOLOGICAL_PATTERNS,
    *SMALL_MOLECULE_PATTERNS,
]
