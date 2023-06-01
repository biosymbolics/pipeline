ACTIONS = [
    "activator",
    "agonist",
    "analog",
    "antagonist",
    "blocker",
    "chaperone",
    "degrader",
    "downregulator",
    "engager",
    "immunomodulator",
    "inducer",
    "inhibitor",
    "modulator",
    "potentiator",
    "stimulator",
    "suppressor",
    "upregulator",
]

BIOLOGIC_TYPES = [
    "antibody",
    "cell therapy",
    "cytokine",
    "enzyme",
    "factor",
    "factor [ivx]{1-3}",
    "Fab(?: region)?",
    "fc[-\\s]fusion(?: protein)?",
    "fusion protein",
    "gene (?:[a-z]+ )?therapy",
    "growth factor",
    "hormone",
    "mrna",
    "peptide",
    "polypeptide",
    "protein",
    "sirna",
    "transcription factor",
    "vaccine",
]

DRUG_CLASS_TYPE = [
    "agent",
    "anti[-]?[a-z].+ agent",
    "anti[-]?[a-z].+s",
    "conjugate",
    "pro[-\\s]?drug",
]

CONJUGATE_TYPES = [
    "antibody[-]?drug conjugate",
    "adc",
    "peptide[-]?drug conjugate",
    "prodrug conjugate",
    "nanoparticle conjugate",
]

IMMUNOTHERAPY_TYPES = [
    "adjuvant",
    "bispecific",
    "car[-]?t",
    "car[-]?nk",
    "chimeric antigen receptor t[-\\s]?cell?",
    "monoclonal antibody",
    "mab",
    "adoptive cell transfer",
    "tumor[-| ]infiltrating lymphocyte",
    "t[-\\s]?cell engager",
    "tce",
]


EFFECTS = [
    *ACTIONS,
    *["dual[-\\s]?" + effect for effect in ACTIONS],
    *["tri[-\\s]?" + effect for effect in ACTIONS],
]


EFFECTS_AND_CLASSES = [
    *BIOLOGIC_TYPES,
    *CONJUGATE_TYPES,
    *EFFECTS,
    *DRUG_CLASS_TYPE,
    *IMMUNOTHERAPY_TYPES,
]
