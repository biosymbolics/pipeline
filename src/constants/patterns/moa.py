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
    "downregulator",
    "engager",
    "immunomodulator",
    "inducer",
    "inhibitor",
    "modulator",
    "potentiator",
    "pro[-\\s]?drug",
    "stimulator",
    "suppressor",
    "upregulator",
]

CONJUGATE_TYPES = [
    "antibody[-]?drug conjugate",
    "adc",
    "peptide[-]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

BIOLOGIC_TYPES = [
    "adoptive cell transfer",
    "antibody",
    "cell therapy",
    "car[-]?t",
    "car[-]?nk",
    "chimeric antigen receptor t[-\\s]?cell?",
    "cytokine",
    "enzyme",
    # "factor",
    "factor [ivx]{1-3}",
    "Fab(?: region)?",
    "fc[-\\s]fusion(?: protein)?",
    "fusion protein",
    "gene (?:[a-z]+ )?therapy",
    "growth factor",
    "hormone",
    "monoclonal antibody",
    "mab",
    "mrna",
    "peptide",
    "polypeptide",
    "protein",
    "sirna",
    "tumor[-| ]infiltrating lymphocyte",
    "t[-\\s]?cell engager",
    "tce",
    "transcription factor",
    "vaccine",
    *CONJUGATE_TYPES,
]

DRUG_CLASS_TYPE = [
    "anti[-]?[a-z].+ agent",
    "anti[-]?[a-z].+s",
]

IMMUNOTHERAPY_TYPES = [
    "adjuvant",
    "bispecific",
]


EFFECTS = [
    *ACTIONS,
    *["dual[-\\s]?" + effect for effect in ACTIONS],
    *["tri[-\\s]?" + effect for effect in ACTIONS],
]


MOA_SUFFIXES = [
    *EFFECTS,
    *BIOLOGIC_TYPES,
]

MOA_INFIXES = [*IMMUNOTHERAPY_TYPES, *DRUG_CLASS_TYPE]
