"""
Patterns related to generic mechanisms of action

TODO: car-ts belong in biologics
"""
from utils.re import ALPHA_CHARS

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
    "enhancer",
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
    "antibody[- ]?drug conjugate",
    "adc",
    "peptide[- ]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

BIOLOGIC_SUFFIXES = [
    "adoptive cell transfer",
    # "adjuvant", # more likely to be indication (e.g. Stage IB-IIIA Adjuvant NSCLC)
    "bi[- ]?specific(?: antibody)?",
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
    "fc[- ]fusion(?: protein)?",
    "fragment",
    "fusion protein",
    f"gene (?:{ALPHA_CHARS(4)}\\s?)?therapy",
    "growth factor",
    "hormone",
    "isoform",
    "ligand",
    "monoclonal antibody",
    "mono[- ]?specific(?: antibody)?",
    "mrna",
    "neoadjuvant",
    "peptide",
    "peri[- ]?adjuvant",
    "polypeptide",
    "protein",
    "sirna",
    "stem cell transplant",
    "substitute",
    "tumor[- ]?infiltrating lymphocyte",
    "t[- ]?cell engager",
    "tce",
    "t[- ]?cell receptor",
    "tcr",
    "t[- ]?cell engaging receptor",
    "tcer",
    "transcription factor",
    "vaccine",
    *CONJUGATE_TYPES,
]

EFFECTS = [
    *ACTIONS,
    *["dual[- ]?" + effect for effect in ACTIONS],
    *["tri[- ]?" + effect for effect in ACTIONS],
    *[f"anti[- ]?{ALPHA_CHARS('+')} {action}" for action in ACTIONS],
]

CAR_T_INFIXES = [
    "car[- ]?t",
    "car[- ]?nk",
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
    *CAR_T_INFIXES,
    ".+-targeted",
    ".+-binding",
]

MOA_SUFFIXES = [
    *EFFECTS,
    *BIOLOGIC_SUFFIXES,
    "composition",  # compound suffix
    "regimen",
    "therapy",
]


MOA_PREFIXES = ["recombinant", "therapeutic", *[f"{action} of" for action in ACTIONS]]
