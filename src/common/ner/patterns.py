"""
NER Patterns (SpaCy)

TODO:
- O-glc-NAcase
- Epiregulin/TGFα MAB
- Pirtobrutinib (LOXO-305)
- LP(a) Inhibitor

"""
MOA_PATTERN = [
    # {"HEAD": { "IN": ["PROPN", "NOUN", "ADJ"] }, "OP": "+"},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "+"},
    {
        "LOWER": {
            "IN": [
                "inhibitor",
                "dual inhibitor",
                "tri-inhibitor",
                "agonist",
                "dual agonist",
                "tri-agonist",
                "antagonist",
                "dual antagonist",
                "tri-antagonist",
                "degrader",
                "vaccine",
                "vaccination",
                "gene therapy",
                "gene transfer",
                "gene transfer therapy",
                "car-t",
                "car t",
                "chimeric antigen receptor t-cell",
                "chimeric antigen receptor t cell",
                "conjugate",
                "monoclonal antibody",
                "mab",
                "antibody",
                "antibody-drug conjugate",
                "adc",
                "tce",
                "T-cell engager",
                "bcma tce",
                "bcma t-cell engager",
                "bispecific",
                "bispecific antibody",
                "bispecific t-cell engager",
                "bte",
                "t cell engaging antibody",
                "t cell engaging antibodies",
                "prodrug",
                "pro-drug",
                "pro drug",
                "sirna",
                "therapeutic",
                "therapeutic agent",
            ]
        },
    },
]

# IN supported with REGEX supported in 3.5+
INVESTIGATIONAL_ID_PATTERN_1 = [
    {
        "TEXT": {"REGEX": "[a-zA-Z]{2,}[-]?[0-9]{3,}"},  # XYZ123, XYZ-123, Xyz-123
    },
]

INVESTIGATIONAL_ID_PATTERN_2 = [
    {
        "TEXT": {"REGEX": "[A-Z]{2,}[- ]?[0-9]{3,}"},  # XYZ 123, XYZ-123"
    },
]

INVESTIGATIONAL_ID_PATTERN_3 = [
    {
        "TEXT": {
            "REGEX": "[A-Z0-9]{2,8}-[A-Z0-9]{2,8}-[A-Z0-9]{2,8}"
        },  # e.g. CAB-AXL-ADC; will miss lower-cased terms
    },
]

# TODO: Preclinical
TRIAL_PHASE_PATTERN = [
    {"LOWER": {"REGEX": "phase (?:[0-4]|i{1,3}|iv)+"}},  # phase 1, phase i, phase I
]

REGULATORY_DESIGNATION_PATTERN = [
    {"LOWER": {"IN": ["fast track", "accelerated approval"]}}
]
