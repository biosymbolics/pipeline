"""
Patterns around other stuff like regulatory and clinical trials
"""

# unused
THERAPEUTIC_AREAS: list[str] = [
    "cardiovascular",
    "hematology",
    "immunology",
    "infectious disease",
    "oncology",
    "neurodegeneration",
    "neurology",
]

# phase 1, phase i, phase I
TRIAL_PHASE_PATTERNS = [
    {"LOWER": "preclinial"},
    {"LOWER": {"REGEX": "phase (?:[0-4]|i{1,3}|iv)+"}},
]

REGULATORY_DESIGNATION_PATTERN = [
    {"LOWER": {"IN": ["fast track", "accelerated approval"]}}
]

OTHER = {
    "LOE": "loss of exclusivity",
}
