"""
Patterns related to generic mechanisms of action

TODO: car-ts belong in biologics
"""
from common.utils.re import ALPHA_CHARS


ACTIONS = [
    "activator",
    "agent",
    "agonist",
    "analog",
    "antagonist",
    "antigen",  # prostate stem cell antigen (PSCA)
    "blocker",
    "blocking antibody",
    "chaperone",
    "conjugate",
    "protein degrader",
    "degrader",
    "down[-\\s]?regulator",
    "engager",
    "immuno[-\\s]?modulator",
    "inducer",
    "inhibitor",
    "ligase modulator",
    "modulator",
    "potentiator",
    "pro[-\\s]?drug",
    "stimulator",
    "suppressor",
    "up[-\\s]?regulator",
]

CONJUGATE_TYPES = [
    "antibody[-\\s]?drug conjugate",
    "adc",
    "peptide[-\\s]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

CAR_T_INFIXES = [
    "car[-\\s]?t",
    "car[-\\s]?nk",
    "chimeric antigen receptor.*",
    "bcma nke",
    "nke",
    "natural killer cells",
    "cd[0-9]{1,2}",
    "cd[0-9]{1,2}-cd[0-9]{2}",
    "cd[0-9]{1,2}xcd[0-9]{2}",  # CD47xCD20
    "cd[0-9]{1,2}x[A-Z]{3,6}",  # CD3xPSCA
    "il[0-9]{1,2}-cd[0-9]{2}",
]

BIOLOGIC_SUFFIXES = [
    "adoptive cell transfer",
    # "adjuvant", # more likely to be indication (e.g. Stage IB-IIIA Adjuvant NSCLC)
    "antibody",
    "bispecific",
    "bispecific antibody",
    "cell therapy",
    "chemotherapy",
    "cytokine",
    "enzyme",
    # "factor",
    "factor [ivx]{1-3}",
    "fab(?: region)?",
    "fc",
    "fc[-\\s]fusion(?: protein)?",
    "fusion protein",
    f"gene (?:{ALPHA_CHARS(4)}\\s?)?therapy",
    "growth factor",
    "hormone",
    "monoclonal antibody",
    "mab",
    "mrna",
    "neoadjuvant",
    "peptide",
    "peri[-\\s]?adjuvant",
    "polypeptide",
    "protein",
    "sirna",
    "stem cell transplant",
    "tumor[-\\s]infiltrating lymphocyte",
    "t[-\\s]?cell engager",
    "tce",
    "t[-\\s]?cell receptor",
    "tcr",
    "t[-\\s]?cell engaging receptor",
    "tcer",
    "transcription factor",
    "vaccine",
    *CONJUGATE_TYPES,
]

DRUG_CLASS_TYPE = [
    f"anti[-\\s]?{ALPHA_CHARS('+')} agent",
    f"anti[-\\s]?{ALPHA_CHARS('+')}s",
    f"anti-{ALPHA_CHARS('+')}",
]

EFFECTS = [
    *ACTIONS,
    *["dual[-\\s]?" + effect for effect in ACTIONS],
    *["tri[-\\s]?" + effect for effect in ACTIONS],
]


MOA_SUFFIXES = [
    *EFFECTS,
    *BIOLOGIC_SUFFIXES,
    "regimen",
]

MOA_INFIXES = [*DRUG_CLASS_TYPE, *CAR_T_INFIXES]
