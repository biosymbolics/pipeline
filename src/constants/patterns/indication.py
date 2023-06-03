"""
Patterns around indications
"""

ONCOLOGY_WORDS = [
    "cancer",
    "carcinoma",
    "hematologic malignancy",
    "leukemia",
    "lymphoma",
    "malignant",
    "melanoma",
    "myeloma",
    "sarcoma",
    "tumor",
]

STAGES = ["I", "II", "III", "IV"]
SUB_STAGES = ["a", "b", "c"]
STAGE_WORDS = [
    *STAGES,
    *[[f"{stage}{sub_stage}" for sub_stage in SUB_STAGES] for stage in STAGES],
]
STAGE_TYPES = ["stage", "grade"]
SEVERITY_WORDS = [
    "relapsed",
    "refactory",
    "metastatic",
    *[[f"{type} {word}" for word in STAGE_WORDS] for type in STAGE_TYPES],
]


INDICATION_MODIFIERS = ["acute", "chronic"]

INDICATION_WORDS = [
    "disease",
    "disorder",
    "condition",
    "syndrome",
    "autoimmune",
    "failure",
    *INDICATION_MODIFIERS,
    *ONCOLOGY_WORDS,
]

TREATMENT_LINE_RE = (
    "(?:{1-4}[ ]?|{1-4}-{1-4})(?:l|line)+?"  # 1L, 2L, 3L, 4L, 1L+, 2L+, 3L+, 4L+
)
