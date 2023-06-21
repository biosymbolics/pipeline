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
    "antibody",
    "antibody constructs",
    "binding molecule",
    "blocker",
    "blocking antibody",
    "chaperone",
    "chemotherapy",
    "compound",
    "conjugate",
    "protein degrader",
    "degrader",
    "degradation",
    "derivative",
    "down regulator",
    "down-regulator",
    "downregulator",
    "engager",
    "immunomodulator",
    "immuno-modulator",
    "immuno modulator",
    "inducer",
    "inhibitor",
    "inhibition",
    "ligase modulator",
    "modulator",
    "potentiator",
    "prodrug",
    "pro-drug",
    "pro drug",
    "stimulator",
    "suppressor",
    "up regulator",
    "up-regulator",
    "upregulator",
]

CONJUGATE_TYPES = [
    "antibody[-\\s]?drug conjugate",
    "adc",
    "peptide[-\\s]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

BIOLOGIC_SUFFIXES = [
    "adoptive cell transfer",
    # "adjuvant", # more likely to be indication (e.g. Stage IB-IIIA Adjuvant NSCLC)
    "bispecific",
    "bispecific antibody",
    "blockade",
    "cells",
    "cell therapy",
    "cell transfer therapy",
    "cytokine",
    "enzyme",
    # "factor",
    "factor [ivx]{1-3}",
    "fab(?: region)?",
    "fc",
    "fc[-\\s]fusion(?: protein)?",
    "fragment",
    "fusion protein",
    f"gene (?:{ALPHA_CHARS(4)}\\s?)?therapy",
    "growth factor",
    "hormone",
    "isoform",
    "ligand",
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
    "substitute",
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

EFFECTS = [
    *ACTIONS,
    *["dual[-\\s]?" + effect for effect in ACTIONS],
    *["tri[-\\s]?" + effect for effect in ACTIONS],
]

DRUG_CLASS_TYPE = [
    f"anti[-\\s]?{ALPHA_CHARS('+')} agent",
    f"anti[-\\s]?{ALPHA_CHARS('+')}s",
    f"anti-{ALPHA_CHARS('+')}",
]

CAR_T_INFIXES = [
    "car[-\\s]?t",
    "car[-\\s]?nk",
    "(?:targeting )?chimeric antigen receptor.*",
    "bcma nke",
    "nke",
    "natural killer(?: cells)?",
    "nkt(?: cells)?",
    "nkc",
    "cd[0-9]{1,2}",
    "cd[0-9]{1,2}-cd[0-9]{2}",
    "cd[0-9]{1,2}xcd[0-9]{2}",  # CD47xCD20
    "cd[0-9]{1,2}x[A-Z]{3,6}",  # CD3xPSCA
    "il[0-9]{1,2}-cd[0-9]{2}",
]

MOA_INFIXES = [
    *DRUG_CLASS_TYPE,
    *CAR_T_INFIXES,
    ".+-targeted",
    ".+-based",
    ".+-binding",
]

MOA_SUFFIXES = [
    *EFFECTS,
    *BIOLOGIC_SUFFIXES,
    "composition",
    "regimen",
    "therapy",
    "variants",
]


MOA_PREFIXES = ["recombinant", "therapeutic", *ACTIONS]
