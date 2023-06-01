from common.utils.re import get_or_re


ACTIONS = [
    "activator",
    "agent",
    "agonist",
    "analog",
    "antagonist",
    "blocker",
    "blocking antibody",
    "chaperone",
    "conjugate",
    "degrader",
    "down-?regulator",
    "engager",
    "immuno-?modulator",
    "inducer",
    "inhibitor",
    "modulator",
    "potentiator",
    "pro[-\\s]?drug",
    "stimulator",
    "suppressor",
    "up-?regulator",
]

CONJUGATE_TYPES = [
    "antibody[-]?drug conjugate",
    "adc",
    "peptide[-]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

CAR_T_SUFFIXES = [
    "car[-]?t",
    "car[-]?nk",
    "chimeric antigen receptor (?:(car) )?t[-\\s]?cell?",
    "chimeric antigen receptor (?:(car) )?natural killer cell",
    "bcma nke",
    "nke",
]

CAR_T_INFIXES = [
    "cd{0-9}{2}",
    "cd{0-9}{2}xcd{0-9}{2}",
]

BIOLOGIC_SUFFIXES = [
    "adoptive cell transfer",
    "adjuvant",
    "antibody",
    "bispecific",
    "bispecific antibody",
    "cell therapy",
    "cytokine",
    "enzyme",
    # "factor",
    "factor [ivx]{1-3}",
    "fab(?: region)?",
    "fc",
    "fc[-\\s]fusion(?: protein)?",
    "fusion protein",
    "gene (?:[a-z]+ )?therapy",
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
    "tumor[-| ]infiltrating lymphocyte",
    "t[-\\s]?cell engager",
    "tce",
    "transcription factor",
    "vaccine",
    f"cd{0-9}{2}.+{get_or_re(CAR_T_SUFFIXES)}",
    *CAR_T_SUFFIXES,
    *CONJUGATE_TYPES,
]

DRUG_CLASS_TYPE = [
    "anti[-]?[a-z].+ agent",
    "anti[-]?[a-z].+s",
    "anti-[a-z0-9]+",
]

EFFECTS = [
    *ACTIONS,
    *["dual[-\\s]?" + effect for effect in ACTIONS],
    *["tri[-\\s]?" + effect for effect in ACTIONS],
]


MOA_SUFFIXES = [
    *EFFECTS,
    *BIOLOGIC_SUFFIXES,
]

MOA_INFIXES = [*DRUG_CLASS_TYPE, *CAR_T_INFIXES]
