"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""
from utils.re import get_or_re

# TODO: maybe choose NCI as canonical name
UMLS_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
    "C0005525": "Modulator",  # Biological Response Modifiers https://uts.nlm.nih.gov/uts/umls/concept/C0005525
    "C1145667": "Binder",  # https://uts.nlm.nih.gov/uts/umls/concept/C1145667
    "C1420201": "SGLT2",  # otherwise SLC5A2
    # "C1706082": "Compound",
    "C1550602": "Additive",  # otherwise "Additive (substance)"
}

UMLS_CUI_SUPPRESSIONS = {
    "C0243083": "associated disease",
    "C3263722": "Traumatic AND/OR non-traumatic injury",
    "C1706082": "Compound",
    "C0009429": "combo",
    "C0596316": "chemical group",
    "C1547776": "substance",
    "C0991538": "orderable drug form",
    "C1517360": "gem 220",
    "C0007992": "pharmacological phenomenon",
    "C1698899": "solid drug form",
    "C0525067": "laboratory chemicals",
    "C1550602": "additive",
    "C1951340": "process",
    "C0013227": "Pharmaceutical Preparations",
    "C1963578": "procedure",
    "C0009429": "combination",
    "C0963641": "cat combination",
    "C0596316": "chemical group",
    "C1173729": "SPES herbal extract",
    "C0233656": "mental condensation",  # matches condensation
    "C0242781": "disease transmission",  # transmission
    "C0012359": "Pathological Dilatation",
    "C0234985": "Mental deterioration",
    "C0000925": "Incised wound",
    "C0332157": "exposure to",
    "C0233601": "Spraying behavior",
    "C0542351": "Battery (assault)",
    "C1514893": "physiologic resolution",
    "C0332568": "Pad Mass",  # pad
    "C0445243": "S-Pouch",
    "C0599156": "Transition Mutation",
    "C1136365": "gene component",  # matches too much stuff
    "C1704681": "gene probe",  # matches 'probe'
    "C0017374": "synthetic gene",  # matches 'synthetic'
}


# suppress UMLS entities matching these names
# assumes closest matching alias would match the suppressed name (sketchy)
# does not support re, and matches based on implicit (?:^|$|\s) (word in name.split(" "))
UMLS_NAME_SUPPRESSIONS = [
    "rat",
    "mouse",
    "categorized",  # tend to be generic categories, e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0729761
    "category",  # https://uts.nlm.nih.gov/uts/umls/concept/C1709248
    "preparations",  # https://uts.nlm.nih.gov/uts/umls/concept/C2267085
    "other",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C3540010
    "and",
    "or",
    "by",  # https://uts.nlm.nih.gov/uts/umls/concept/C0682899
    "and/or",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C1276307
    "miscellaneous",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0301555
]


UMLS_COMPOUND_TYPES = {
    "T103": "Chemical",
    "T104": "Chemical Viewed Structurally",
    "T109": "Organic Chemical",
    # "T120": "Chemical Viewed Functionally", # mech??
    "T121": "Compound",  # "Pharmacologic Substance",  # in mech too
    "T127": "Vitamin",
    # "T167": "Substance",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",
}

UMLS_GENE_PROTEIN_TYPES = {
    "T028": "Gene or Genome",
    "T116": "Amino Acid, Peptide, or Protein",
}

UMLS_BIOLOGIC_TYPES = {
    **UMLS_GENE_PROTEIN_TYPES,
    "T038": "Biologic Function",
    "T043": "Cell Function",
    "T044": "Molecular Function",
    "T045": "Genetic Function",
    "T085": "Molecular Sequence",
    "T086": "Nucleotide Sequence",
    "T087": "Amino Acid Sequence",
    "T088": "Carbohydrate",  #  "Carbohydrate Sequence",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T129": "Immunologic Factor",
    "T192": "Receptor",
}


# TODO: move to bio????
UMLS_MECHANISM_TYPES = {
    # "T041": "Mental Process",
    "T120": "Mechanism",  # "Chemical Viewed Functionally",
    "T121": "Pharmacologic Substance",  # TODO: here or in compound???
    "T123": "Pharmacologic effect",  # "Biologically Active Substance",  # e.g. inhibitor, agonist, antagonist
    "T195": "Antibiotic",
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
    # "T060": "Diagnostic Procedure", # (in diagnostic)
    "T059": "Laboratory Procedure",
    "T061": "Therapeutic or Preventive Procedure",
}

UMLS_INTERVENTION_TYPES = {
    **UMLS_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_DEVICE_TYPES,
    **UMLS_PROCEDURE_TYPES,
    "T168": "Food",
}

UMLS_PATHOGEN_TYPES = {
    "T001": "Organism",  # includes "pathogenic organism"
    "T004": "Fungus",
    "T005": "Virus",
    "T007": "Bacterium",
    "T204": "Eukaryote",  # includes parasites
}

UMLS_DISEASE_TYPES = {
    "T019": "Congenital Abnormality",
    "T020": "Acqjuired Abnormality",
    "T037": "Injury or Poisoning",
    "T046": "Pathology",  # "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T050": "Experimental Model of Disease",
    # "T091": "Occupation", # can include theurapeutic areas
    "T184": "Symptom",  # "Sign or Symptom",
    "T190": "Anatomical Abnormality",
    "T191": "Cancer",  # "Neoplastic Process",  # e.g. 'Mantle cell lymphoma'
    **UMLS_PATHOGEN_TYPES,
}


UMLS_PHENOTYPE_TYPES = {
    "T022": "Body System",
    # "T023": "Body Part, Organ, or Organ Component", # matches too many things, e.g. "10" for some tooth thing
    "T024": "Tissue",
    "T025": "Cell",
    "T026": "Cell Component",
    "T028": "Gene",  # "Gene or Genome",
    "T031": "Body Substance",  # includes plaque, atherosclerotic
    "T033": "Finding",
    "T042": "Organ or Tissue Function",  # includes "graft rejection"
    "T067": "Process",  # "Phenomenon or Process"
    # "T101": "Patient or Disabled Group",
}

UMLS_DIAGNOSTIC_TYPES = {
    "T034": "laboratory or test result",
    "T060": "Diagnostic Procedure",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
}

UMLS_RESEARCH_TYPES = {"T063": "research activity"}

UMLS_OTHER_TYPES = {
    # "T068": "Human-caused Phenomenon or Process",  # ??
    "T122": "biomedical or dental material",
    "T131": "Hazardous Substance",  # "Hazardous or Poisonous Substance",
    "T196": "Element",  # "Element, Ion, or Isotope",
    "T049": "Molecular Dysfunction",  # "Cell or Molecular Dysfunction" # e.g. DNA Strand Break
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
}


# most preferred is use-case specific, i.e. we're most interested in compounds > procedures and devices
MOST_PREFERRED_UMLS_TYPES = {
    **UMLS_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_DISEASE_TYPES,
}


BIOSYM_UMLS_TFIDF_PATH = (
    "https://biosym-umls-tfidf.s3.amazonaws.com/tfidf_vectorizer.joblib"
)
