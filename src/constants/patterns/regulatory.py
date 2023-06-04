"""
Patterns around regulatory and clinical trials
"""

# phase 1, phase i, phase I
TRIAL_PHASE_PATTERNS = [
    {"LOWER": "preclinial"},
    {"LOWER": {"REGEX": "phase (?:[0-4]|i{1,3}|iv)+"}},
]

REGULATORY_DESIGNATION_PATTERN = [
    {"LOWER": {"IN": ["fast track", "accelerated approval"]}}
]
