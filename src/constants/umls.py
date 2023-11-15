"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""

# TODO: maybe choose NCI as canonical name
UMLS_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
    "C0005525": "Modulator",  # Biological Response Modifiers https://uts.nlm.nih.gov/uts/umls/concept/C0005525
    "C1145667": "Binder",  # https://uts.nlm.nih.gov/uts/umls/concept/C1145667
}

# suppress UMLS entities matching these names
# assumes closest matching alias would match the suppressed name (sketchy)
# does not support re, and matches based on implicit (?:^|$|\s) (word in name.split(" "))
UMLS_NAME_SUPPRESSIONS = set(
    [
        ", rat",
        ", mouse",
        "categorized",  # tend to be generic categories, e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0729761
        "category",  # https://uts.nlm.nih.gov/uts/umls/concept/C1709248
        "preparations",  # https://uts.nlm.nih.gov/uts/umls/concept/C2267085
        "used in",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C3653437
        "for treatment",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C3540759
        "other",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C3540010
        "and",
        "or",
        "and/or",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C1276307
        "miscellaneous",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0301555
    ]
)


UMLS_COMPOUND_TYPES = {
    "T103": "Chemical",
    "T104": "Chemical Viewed Structurally",
    "T109": "Organic Chemical",
    "T120": "Chemical Viewed Functionally",
    "T121": "Pharmacologic Substance",
    "T122": "biomedical or dental material",
    "T123": "Biologically Active Substance",
    "T127": "Vitamin",
    "T167": "Substance",
    "T195": "Antibiotic",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",
}


UMLS_BIOLOGIC_TYPES = {
    "T043": "cell function",
    "T028": "Gene or Genome",
    "T085": "Molecular Sequence",
    "T086": "Nucleotide Sequence",
    "T087": "Amino Acid Sequence",
    "T088": "Carbohydrate Sequence",
    "T192": "Receptor",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T116": "Amino Acid, Peptide, or Protein",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T129": "Immunologic Factor",
}


UMLS_MECHANISM_TYPES = {
    "T038": "Biologic Function",
    "T041": "Mental Process",
    "T044": "Molecular Function",
    "T045": "Genetic Function",
    "T123": "Biologically Active Substance",  # e.g. inhibitor, agonist, antagonist
}

# not necessarily all interventions.
UMLS_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_COMPOUND_TYPES,
    **UMLS_BIOLOGIC_TYPES,
    **UMLS_MECHANISM_TYPES,
}

UMLS_DEVICE_TYPES = {
    "T074": "Medical Device",
    "T075": "Research Device",
    "T203": "Drug Delivery Device",
}

UMLS_PROCEDURE_TYPES = {
    # "T058": "Health Care Activity",
    "T059": "Laboratory Procedure",
    "T060": "Diagnostic Procedure",
    "T061": "Therapeutic or Preventive Procedure",
}

UMLS_INTERVENTION_TYPES = {
    **UMLS_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_DEVICE_TYPES,
    **UMLS_PROCEDURE_TYPES,
    "T168": "food",
}

UMLS_DISEASE_TYPES = {
    "T019": "Congenital Abnormality",
    "T020": "Acqjuired Abnormality",
    "T037": "Injury or Poisoning",
    "T046": "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T049": "Cell or Molecular Dysfunction",
    "T184": "Sign or Symptom",
    "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process",  # e.g. 'Mantle cell lymphoma'
}


UMLS_PHENOTYPE_TYPES = {
    "T022": "Body System",
    "T023": "Body Part, Organ, or Organ Component",
    "T024": "Tissue",
    "T025": "Cell",
    "T026": "cell component",
    "T028": "Gene or Genome",
    "T033": "Finding",
    "T067": "Phenomenon or Process",
    "T101": "Patient or Disabled Group",
}


UMLS_DIAGNOSTIC_TYPES = {
    "T034": "laboratory or test result",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
}

UMLS_RESEARCH_TYPES = {"T063": "research activity"}

UMLS_OTHER_TYPES = {
    # "T068": "Human-caused Phenomenon or Process",  # ??
    "T131": "Hazardous or Poisonous Substance",
    "T196": "Element, Ion, or Isotope",
}

UMLS_PATHOGEN_TYPES = {
    "T004": "Fungus",
    "T005": "Virus",
    "T007": "Bacterium",
}


# TODO: all these names are lacking distinctness
BIOMEDICAL_GRAPH_UMLS_TYPES = {
    **UMLS_INTERVENTION_TYPES,
    **UMLS_DISEASE_TYPES,
    **UMLS_DIAGNOSTIC_TYPES,
    **UMLS_RESEARCH_TYPES,
    **UMLS_OTHER_TYPES,
}

PREFERRED_UMLS_TYPES = {
    **BIOMEDICAL_GRAPH_UMLS_TYPES,
    **UMLS_PHENOTYPE_TYPES,
    **UMLS_PATHOGEN_TYPES,
}

# most preferred is use-case specific, i.e. we're most interested in compounds > procedures and devices
MOST_PREFERRED_UMLS_TYPES = {
    **UMLS_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_DISEASE_TYPES,
}
