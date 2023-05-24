"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""
UMLS_COMPOUND_TYPES = {
    "T109": "Organic Chemical",
    "T121": "Pharmacologic Substance",
    "T127": "Vitamin",
    "T167": "Substance",
    "T195": "Antibiotic",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",
}

UMLS_BIOLOGIC_TYPES = {
    "T086": "Nucleotide Sequence",
    "T087": "Amino Acid Sequence",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T116": "Amino Acid, Peptide, or Protein",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T129": "Immunologic Factor",
}

UMLS_MECHANSIM_TYPES = {
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
    **UMLS_MECHANSIM_TYPES,  # trying to get MoAs
}

UMLS_DEVICE_TYPES = {
    "T074": "Medical Device",
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
}

UMLS_CONDITION_TYPES = {
    "T019": "Congenital Abnormality",
    "T037": "Injury or Poisoning",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process",  # e.g. 'Mantle cell lymphoma'
}

UMLS_PHENOTYPE_TYPES = {
    "T022": "Body System",
    "T024": "Tissue",
    "T025": "Cell",
    "T028": "Gene or Genome",
    "T033": "Finding",
    "T046": "Pathologic Function",
    "T067": "Phenomenon or Process",
    "T101": "Patient or Disabled Group",
    "T184": "Sign or Symptom",
}

UMLS_INDICATION_TYPES = {
    **UMLS_CONDITION_TYPES,
    **UMLS_PHENOTYPE_TYPES,
}


UMLS_OTHER_TYPES = {
    "T068": "Human-caused Phenomenon or Process",  # ??
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T131": "Hazardous or Poisonous Substance",
}
