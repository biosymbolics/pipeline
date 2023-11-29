"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""

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
    "C1440188": "C little e",  # matches too much stuff
    "C0313108": "Blood group antibody big C little e",
    "C0243083": "associated disease",
    "C3263722": "Traumatic AND/OR non-traumatic injury",
    "C1706082": "Compound",
    "C0009429": "combo",
    "C0596316": "chemical group",
    "C1547776": "substance",
    "C0991538": "orderable drug form",
    "C0044729": "compound A",
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
    "C4293691": "abnormality of digestive system morphology",
    "C0268275": "AB variant",  # Tay-Sachs AB variant
    "C1173729": "SPES herbal extract",
    "C0031516": "pheromone",  # matches scented
    "C0332837": "traumatic implant",  # matches implant
    "C0037188": "Sinoatrial Block ",  # matches block
    "C0027627": "neoplasm metastasis",  # matches "secondary"
    "C0011175": "dehydration",  # matches "dehydrated"
    "C0015967": "fever",  # matches high temp
    "C0151747": "renal tubular disorder",  # TUBULAR
    "C0026837": "muscle rigidity",  # matches rigid
    "C0700198": "Pulmonary aspiration",  # matches aspiration
    "C0233656": "mental condensation",  # matches condensation
    "C0043242": "Superficial abrasion",  # matches abrasion
    "C0332875": "Congenital webbing",  # matches web
    "C1510411": "metaplastic cell transformation",  # matches transformation
    "C2926602": "discharge",  # discharge, drainage
    "C0242781": "disease transmission",  # transmission
    "C0220811": "consumption (TB)",  # consumption
    "C4074771": "Sterility, Reproductive",
    "C1572199": "COBRA MENS PERFORMANCE ENHANCER",
    "C0158328": "Trigger Finger Disorder",  #  (trigger)
    "C0011119": "Decompression Sickness",  # (decompression, bends)
    "C0555975": "sore bottom",
    "C0877578": "treatment related secondary malignancy",  # (secondary)
    "C0856151": "fat redistribution",  # (redistribution)
    "C0263557": "saddle sore",
    "C0276640": "Transmissible mink encephalopathy",  # tem
    "C0858714": "bone fragile",
    "C0595920": "accomodation",  # pathological
    "C1318484": "Chimeria disorder",  # (chimeria)
    # "C0020517": "Hypersensitivity", # (allergy, sensitivity)
    "C0020488": "Hypernatremia",  # (na excess)
    "C0278134": "Absence of sensation",  # (anaesthesia)
    "C0220724": "CONSTRICTING BANDS, CONGENITAL",  # "ABS",
    "C2029593": "elongated head",
    "C0265604": "mirror hands",
    "C0012359": "Pathological Dilatation",
    "C0234985": "Mental deterioration",
    "C0027960": "Nevus",
    "C1384489": "Scratch marks",
    "C0277825": "Bone sequestrum",
    "C0392483": "Embedded teeth",
    "C0010957": "Tissue damage",
    "C0015376": "Extravasation",
    "C0011389": "Dental Plaque",
    "C0597561": "temperature sensitive mutant",
    "C0036974": "Shock",
    "C0233494": "Tension",
    "C0011334": "Dental caries",
    "C0013146": "Drug abuse",
    "C0030201": "Pain, Postoperative",
    "C0000925": "Incised wound",
    "C0332157": "exposure to",
    "C0151908": "Dry skin",
    "C0024524": "Malacia",
    "C0037293": "Skin tag",
    "C0233601": "Spraying behavior",
    "C4023747": "Abnormal curve of the spine",
    "C0262568": "Subendocardial myocardial infarction",  # semi
    "C0542351": "Battery (assault)",
    "C0036572": "Seizures",
    "C0858950": "Mental aberration",
    "C0001511": "Tissue Adhesions",
    "C0080194": "Muscle strain",
    "C1514893": "physiologic resolution",
    "C0003516": "Aortopulmonary Septal Defect",  # ap window
    "C0332568": "Pad Mass",  # pad
    "C0445243": "S-Pouch",
    "C0349506": "Photosensitivity of skin",
    "C0599156": "Transition Mutation",
    "C0033119": "Puncture wound",  # puncture, prick, pricks
    "C4721411": "Osteolysis",  # dissolution
    "C3658343": "genes, microbial",  # matches too much stuff
    "C1136365": "gene component",  # matches too much stuff
    "C1704681": "gene probe",  # matches 'probe'
    "C0017374": "synthetic gene",  # matches 'synthetic'
    "C1334103": "IL13",  # matches 'allergic rhinitis' etc; ok to leave the match to "Interleukin-13"
    "C0017345": "genes, fungal",
    "C0017343": "genes, env",  # matches envelope
    "C0678941": "mutant gene",  # matches variation
    "C0443640": "Specific antibody",
    # "C1516451": "chemical modifier",
}

# "C1425681": "RTTN",  # matches 'rotatin' sigh
# "C1424156": "TRIM9",  # matches trim
# "C1823381": "TMEM121",  # matches hole
# "C1428870": "SH3YL1",  # matches 'ray'
# "C1825598": "impact Gene",
# "C2239865": "PUS10",  # matches "downstream"
# "C3470573": "SMIM10L2A",  # matches LEDs
# "C1413357": "CFLAR",  # matches flame
# "C1366450": "BAD gene",  # matches bad
# "C0919453": "cancer susceptibility gene",  # matches susceptibility
# "C1367342": "TERT",  # matches 'tert'
# "C1336558": "TACC3",  # matches masking
# "C1824612": "CCDC86",  # matches cyclone
# "C1336594": "TBPL1",  # matches stud
# "C1414372": "ELAVL2",  # matches HUB
# "C1424276": "HM13",  # matches spp.
# "C1823652": "VWA1",  # matches warp
# "C1367578": "AR gene",  # spinal and bulbar muscular atrophy
# CD40LG
# Genes, Processed
# PTTG1 securing
# AP5M1 == mud
# EMILIN1 == EMI
# AAVS1 == AAV
# DNAI7 susceptibility
# LITAF simple
# SLURP1 component B
# DNAAF6 twisted
# CD96 tactile

# suppress UMLS entities matching these names
# assumes closest matching alias would match the suppressed name (sketchy)
# does not support re, and matches based on implicit (?:^|$|\s) (word in name.split(" "))
UMLS_NAME_SUPPRESSIONS = [
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
    "by",  # https://uts.nlm.nih.gov/uts/umls/concept/C0682899
    "and/or",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C1276307
    "miscellaneous",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0301555
]

UMLS_COMPOUND_TYPES = {
    "T103": "Chemical",
    "T104": "Chemical Viewed Structurally",
    "T109": "Organic Chemical",
    # "T120": "Chemical Viewed Functionally", # mech??
    "T121": "Pharmacologic Substance",  # in mech too
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
    "T043": "cell function",
    "T044": "Molecular Function",
    "T045": "Genetic Function",
    "T085": "Molecular Sequence",
    "T086": "Nucleotide Sequence",
    "T087": "Amino Acid Sequence",
    "T088": "Carbohydrate Sequence",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T129": "Immunologic Factor",
    "T192": "Receptor",
}


# TODO: move to bio????
UMLS_MECHANISM_TYPES = {
    # "T041": "Mental Process",
    "T120": "Chemical Viewed Functionally",
    "T121": "Pharmacologic Substance",  # TODO: here or in compound???
    "T123": "Biologically Active Substance",  # e.g. inhibitor, agonist, antagonist
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
    "T168": "food",
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
    "T046": "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T050": "Experimental Model of Disease",
    # "T091": "Occupation", # can include theurapeutic areas
    "T184": "Sign or Symptom",
    "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process",  # e.g. 'Mantle cell lymphoma'
    **UMLS_PATHOGEN_TYPES,
}


UMLS_PHENOTYPE_TYPES = {
    "T022": "Body System",
    # "T023": "Body Part, Organ, or Organ Component", # matches too many things, e.g. "10" for some tooth thing
    "T024": "Tissue",
    "T025": "Cell",
    "T026": "cell component",
    "T028": "Gene or Genome",
    "T031": "Body Substance",  # includes plaque, atherosclerotic
    "T033": "Finding",
    "T042": "Organ or Tissue Function",  # includes "graft rejection"
    "T067": "Phenomenon or Process",
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
    "T131": "Hazardous or Poisonous Substance",
    "T196": "Element, Ion, or Isotope",
    "T049": "Cell or Molecular Dysfunction",  # e.g. DNA Strand Break
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
