"""
Patterns around indications
"""

from pydash import flatten


ONCOLOGY_REGEXES: list[str] = [
    "cancer",
    "hematologic malignancy",
    "leukemia",
    "malignant",
    r"\w+?oma",  # neuroblastoma, etc
    "solid tumor",
    "tumor",
]

STAGES: list[str] = ["I", "II", "III", "IV", "1", "2", "3", "4"]
SUB_STAGES: list[str] = ["a", "b", "c"]
STAGE_WORDS: list[str] = [
    *STAGES,
    *flatten([[f"{stage}{sub_stage}" for sub_stage in SUB_STAGES] for stage in STAGES]),
]
STAGE_TYPES: list[str] = ["stage", "grade"]
SEVERITY_REGEXES: list[str] = [
    "accelerated",
    "advanced",
    "anaplastic",
    "blast phase",
    "brittle",
    "castration[-\\s]resistant",
    "end[-\\s]stage",
    "locally advanced",
    "metastatic",
    "mild",
    "moderate",
    "moderate[-\\s]to[-\\s]severe",
    "newly[-\\s]diagnosed",
    "progressive",
    "relapsed",
    "relapsed(?:\\/| or )refractory",
    "refactory",
    "recurrent",
    "relapsing",
    "relapsing[-\\s]remitting",
    "severe",
    "systemic",
    "treatment[-\\s]naive",
    "unresectable",
    *flatten(
        [[f"{type} {word}(?:-{word})?" for word in STAGE_WORDS] for type in STAGE_TYPES]
    ),
]

LINE_RE = "(?:[1-4](?:st|nd|rd|th)?|(?:first|second|third|forth))"
FULL_LINE_RE = LINE_RE + "\\s?" + "(?:l|line)" + "\\+?"
TREATMENT_LINE_RE: str = f"((?:{FULL_LINE_RE})|(?:(?:{LINE_RE}|{FULL_LINE_RE})-{FULL_LINE_RE}))"  # 1L, 1L+, 1st line, first line, 1L-2L, 1L-2L+

INDICATION_MODIFIER_REGEXES: list[str] = [
    "acute",
    "chronic",
    "primary",
    "secondary",
    *SEVERITY_REGEXES,
    TREATMENT_LINE_RE,
    "adjuvant",
]

INDICATION_REGEXES: list[str] = [
    "disease",
    "disorder",
    "syndrome",
    "autoimmune",
    "failure",
    *ONCOLOGY_REGEXES,
]
