"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""

from typing import Literal
from prisma.enums import BiomedicalEntityType

LegacyDomainType = Literal[
    "compounds",
    "biologics",
    "devices",
    "diagnostics",
    "diseases",
    "procedures",
    "mechanisms",
]

# TODO: maybe choose NCI as canonical name
# rewrites preferred name
UMLS_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
    "C0005525": "Modulator",  # Biological Response Modifiers https://uts.nlm.nih.gov/uts/umls/concept/C0005525
    "C1145667": "Binder",  # https://uts.nlm.nih.gov/uts/umls/concept/C1145667
    "C1420201": "SGLT2",  # otherwise SLC5A2
    # "C1706082": "Compound",
    "C1550602": "Additive",  # otherwise "Additive (substance)"
    "C1292856": "Stimulator",  # https://uts.nlm.nih.gov/uts/umls/concept/C1292856 Stimulation procedure
    "C0025080": "Device",  # vs "Medical Device"
    "C0348078": "Form",  # qualitative form
    "C0005515": "Biological Factors",
}

# sets canonical based on (single!) word
UMLS_WORD_OVERRIDES = {
    "modulator": "C0005525",  # "Biological Response Modifiers"
    "modulators": "C0005525",
    "binder": "C1145667",  # binding action
    "binders": "C1145667",  # binding action
    "binding": "C1145667",  # binding action
    "inhibitor": "C1999216",  # https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "inhibitors": "C1999216",  # https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "antagonist": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "antagonists": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
}

UMLS_CUI_ALIAS_SUPPRESSIONS = {
    "C1843354": ["base"],  # "bpifa4p",  # matches base
    "C1427122": ["tube"],  # "tube1",  # matches tube
    "C1823381": ["hole"],  # TMEM121 - hole
    "C0035287": ["re system"],  # "Reticuloendothelial System"
    "C0025611": ["ice", "speed", "crystal"],  # "methamphetamine"
    "C0017374": ["synthetic gene"],  # matches synthetic gene
    "C0044729": ["a compound"],  # "11-dehydrocorticosterone"
    "C0023172": [
        "le cell",
        "le cells",
        "cells, le",
        "cell, le",
        "cells le",
        "cell le",
    ],
    "C0242781": ["transmission"],  # "disease transmission"
    "C0010124": ["compound b", "b compounds"],  # "corticosterone"
    "C1421222": ["connectin"],  # TTN gene
    "C1418824": ["inhibitor 1", "i1"],  # PPP1R1A
    "C0019167": ["e antigens", "e antigen", "antigen e"],  # "Hepatitis B e Antigens
    "C1448132": ["ir"],
    "C0666364": ["odf"],  # TRANCE protein
    "C0022709": ["ace"],  # Peptidyl-Dipeptidase A
    "C5441529": ["er"],  # Estrogen Receptor Family, human
    "C0072402": ["stk"],  # Protein-Serine-Threonine Kinases # ??
    "C0007578": ["cam"],  # cell-adhesion molecule
    "C0069515": ["neu"],  # erbB-2 Receptor
}

UMLS_CUI_SUPPRESSIONS = {
    "C1704222": "genome encoded entity",
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
    "C0012634": "disease",  # useless
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
    "C0597357": "receptor",  # TODO: we don't want to suppress this so much as make it unimportant
    "C0243076": "antagonists",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "C0243192": "agonists",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C2987634
    "C0243077": "inhibitors",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "C0243072": "derivative",  # useless
    "C1413336": "cel gene",  # matches cell
    "C0815040": "acidic amino acid",  # matches amino acid
    "C0008109": "chimera",
    "C0043335": "xenobiotic",
    "C1257890": "population group",  # matches group
    "C0443933": "Sjogren's syndrome B antibody",  # matches monoclonal antibody (not necessary if semantic matching)
    "C1254351": "pharmacological Substance",
    "C0544791": "Inflammatory fistula",
    "C4085054": "particl",  # matches particle
    "C0596611": "Gene Mutation",
    "C0543419": "Sequela of disorder",
    "C0033684": "proteins",  # fix ancestory, get rid of this.
    "C1335532": "protein family",
    "C1148560": "molecular_function",
    "C0012854": "DNA",  # fix ancestory, get rid of this.
    "C0450442": "Agent",
    # "C5235658": "Targeted therapy agent",  # maybe re-enable; it has some utility.
    "C0079429": "gene a",
    "C1260969": "ring device",
    "C0282636": "cell respiration",  # matches respiration
    "C0456386": "Medicament",
    "C1167622": "binding",
    "C0019047": "abnormal hemoglobin",
    "C1516451": "chemical modifier",
    # "C1709058": "modified release dosage form",
    "C2699893": "Molecular Targeted Therapy",  # ??
    "C0004793": "base sequence",  # ??
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
    "product",  # e.g. gene product / https://uts.nlm.nih.gov/uts/umls/concept/C3828300
    "wt",  # wt allele (prefer gene record)
    "mesh",  # e.g. MeSH Russian
    "headings",  # e.g. Medical Subject Headings Norwegian
    "category",  # e.g. Chemicals and Drugs (MeSH Category)
    "schedule",  # e.g. schedule II opium and derivatives
    "affecting",  # C2697974 Agent Affecting Respiratory System
    "drugs",  # e.g. CARDIOVASCULAR SYSTEM DRUGS
    "various",  #  VARIOUS DRUG CLASSES IN ATC
    "unspecified",  # e.g. Nitrogen Compounds, Unspecified
]


UMLS_COMPOUND_TYPES = {
    "T103": "Chemical",
    "T104": "Chemical Viewed Structurally",
    "T109": "Organic Chemical",
    "T121": "Pharmacologic Substance",
    "T127": "Vitamin",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",
}

UMLS_FORMULATION_TYPES = {
    "T122": "biomedical or dental material",  # e.g. Wetting Agents, Tissue Scaffolds
}
UMLS_MAYBE_FORMULATION_TYPES: dict[str, str] = {}


UMLS_MAYBE_COMPOUND_TYPES = {
    "T167": "Substance",
    **UMLS_FORMULATION_TYPES,
}


UMLS_GENE_PROTEIN_TYPES = {
    "T028": "Gene or Genome",
    "T116": "Amino Acid, Peptide, or Protein",
}

UMLS_TARGET_TYPES = {
    **UMLS_GENE_PROTEIN_TYPES,
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",  # 14666 of 14889 are NOT also in UMLS_GENE_PROTEIN_TYPES
}

UMLS_BIOLOGIC_TYPES = {
    **UMLS_TARGET_TYPES,
    "T038": "Biologic Function",  # activation, inhibition, induction, etc
    "T043": "Cell Function",  # regulation, inhibition, transport, secretion, transduction
    "T044": "Molecular Function",  # lots of XYZ process, XYZ activity. also "Opioid mu-Receptor Agonists" (MECHANISM!)
    "T045": "Genetic Function",  # lots of integration, methylation, acetylation, transcription, etc
    # "T085": "Molecular Sequence", # this has 12 entries...
    "T086": "Nucleotide Sequence",  # SNPs, repeats, codons, etc (< 500 entries)
    "T087": "Amino Acid Sequence",  # domains, motifs (200 entries)
    "T088": "Carbohydrate",  #  "Carbohydrate Sequence",
    "T129": "Immunologic Factor",  # proteins, isoforms, enzymes (usually duplicated elsewhere, e.g. "amino acid, peptide, protein")
    # the following could be target types, but avoiding overlap
    "T125": "Hormone",  # 2039 of 3348 are NOT also in UMLS_GENE_PROTEIN_TYPES
    "T126": "Enzyme",  # only 28 or 30924 are NOT also in UMLS_GENE_PROTEIN_TYPES
    "T192": "Receptor",  # only 198 of 5655 are NOT also in UMLS_GENE_PROTEIN_TYPES
}

UMLS_MAYBE_BIOLOGIC_TYPES = {
    "T022": "Body System",
    "T024": "Tissue",
    "T025": "Cell",
    "T026": "Cell Component",
}


UMLS_MECHANISM_TYPES = {
    "T120": "Mechanism",  # "Chemical Viewed Functionally",
    "T121": "Pharmacologic Substance",  # in compount too, but has a lot of important MoAs
    "T123": "Pharmacologic effect",  # "Biologically Active Substance",  # e.g. inhibitor, agonist, antagonist
    "T195": "Antibiotic",
    "T044": "Molecular Function",  # lots of XYZ process, XYZ activity. also "Opioid mu-Receptor Agonists". Incompletely dups "Pharmacologic Substance"
}

UMLS_MAYBE_MECHANISM_TYPES: dict[str, str] = {}


UMLS_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_COMPOUND_TYPES,
    **UMLS_BIOLOGIC_TYPES,
    **UMLS_MECHANISM_TYPES,
}

# maybe / on-the-fence pharmacologic intervention types
UMLS_MAYBE_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_MAYBE_COMPOUND_TYPES,
    **UMLS_MAYBE_BIOLOGIC_TYPES,
    **UMLS_MAYBE_MECHANISM_TYPES,
}

UMLS_DEVICE_TYPES = {
    "T074": "Medical Device",
    "T075": "Research Device",
    "T203": "Drug Delivery Device",
}
UMLS_MAYBE_DEVICE_TYPES = {"T073": "Manufactured Object"}


UMLS_PROCEDURE_TYPES = {
    "T058": "Health Care Activity",
    "T059": "Laboratory Procedure",
    "T061": "Therapeutic or Preventive Procedure",
}
UMLS_MAYBE_PROCEDURE_TYPES: dict[str, str] = {}

UMLS_NON_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_DEVICE_TYPES,
    **UMLS_PROCEDURE_TYPES,
    "T168": "Food",
}

UMLS_INTERVENTION_TYPES = {
    **UMLS_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_NON_PHARMACOLOGIC_INTERVENTION_TYPES,
}

UMLS_MAYBE_NON_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_MAYBE_DEVICE_TYPES,
    **UMLS_MAYBE_PROCEDURE_TYPES,
}

UMLS_MAYBE_INTERVENTION_TYPES = {
    **UMLS_MAYBE_PHARMACOLOGIC_INTERVENTION_TYPES,
    **UMLS_MAYBE_NON_PHARMACOLOGIC_INTERVENTION_TYPES,
}

UMLS_PATHOGEN_TYPES = {
    "T004": "Fungus",
    "T005": "Virus",
    "T007": "Bacterium",
    "T204": "Eukaryote",  # includes parasites
}

UMLS_CORE_DISEASE_TYPES = {
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T191": "Cancer",  # "Neoplastic Process",  # e.g. 'Mantle cell lymphoma'
}

UMLS_NON_PATHOGEN_DISEASE_TYPES = {
    **UMLS_CORE_DISEASE_TYPES,
    "T019": "Congenital Abnormality",
    "T020": "Acqjuired Abnormality",
    "T037": "Injury or Poisoning",
    "T046": "Pathology",  # "Pathologic Function",
    "T050": "Experimental Model of Disease",
    # "T091": "Occupation", # can include theurapeutic areas
    "T184": "Symptom",  # "Sign or Symptom",
    "T190": "Anatomical Abnormality",
}

UMLS_DISEASE_TYPES = {
    **UMLS_NON_PATHOGEN_DISEASE_TYPES,
    **UMLS_PATHOGEN_TYPES,
}

UMLS_PHENOTYPE_TYPES = {
    "T031": "Body Substance",  # includes plaque, atherosclerotic, Amniotic Fluid (bad)
    "T033": "Finding",  # includes Hypotension, Tachycardia, Overweight but a lot of junk too, e.g. retraction (finding)
    "T042": "Organ or Tissue Function",  # includes "graft rejection", but also "Natural regeneration"
    # "T101": "Patient or Disabled Group",
}


# umls types that might be diseases
UMLS_MAYBE_DISEASE_TYPES = {
    **UMLS_PHENOTYPE_TYPES,
    "T040": "Organism Function",  # Positive Regulation of Angiogenesis
    "T041": "Mental Process",
    "T049": "Molecular Dysfunction",  # "Cell or Molecular Dysfunction" # e.g. DNA Strand Break
    "T039": "Physiologic Function",  # e.g. Menopause
    # "T032": "Organism Attribute" # Multi-Drug Resistance, Hair Color
    "T131": "Hazardous Substance",  # "Hazardous or Poisonous Substance",
}


UMLS_DIAGNOSTIC_TYPES = {
    "T034": "laboratory or test result",
    "T060": "Diagnostic Procedure",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
}

UMLS_MAYBE_DIAGNOSTIC_TYPES: dict[str, str] = {}

UMLS_RESEARCH_TYPES = {
    "T062": "Research Activity",
    "T063": "research activity",
}
UMLS_MAYBE_RESEARCH_TYPES: dict[str, str] = {}

UMLS_OTHER_TYPES = {
    # "T068": "Human-caused Phenomenon or Process",  # ??
    "T196": "Element",  # "Element, Ion, or Isotope",
    "T169": "Functional Concept",
    "T067": "Process",  # "Phenomenon or Process" includes Emergency Situation, dehydrogenation
    # "T001": "Organism",  # includes "pathogenic organism" -could be disease or biologic?
    "T080": "Qualitative Concept",  # 23313
    "T098": "Population Group",  # 9664 pregnant women, healthy volunteeers, living donors
    "T082": "Spatial Concept",
}

# UMLS ents of these types will be included in the UMLS load
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


# wider net UMLS types
UMLS_EXTENDED_DISEASE_TYPES = {**UMLS_DISEASE_TYPES, **UMLS_MAYBE_DISEASE_TYPES}
UMLS_EXTENDED_COMPOUND_TYPES = {**UMLS_COMPOUND_TYPES, **UMLS_MAYBE_COMPOUND_TYPES}
UMLS_EXTENDED_BIOLOGIC_TYPES = {
    **UMLS_BIOLOGIC_TYPES,
    **UMLS_MAYBE_BIOLOGIC_TYPES,
}
UMLS_EXTENDED_PHARMACOLOGIC_INTERVENTION_TYPES = {
    **UMLS_EXTENDED_COMPOUND_TYPES,
    **UMLS_EXTENDED_BIOLOGIC_TYPES,
    **UMLS_MECHANISM_TYPES,
}
UMLS_EXTENDED_DEVICE_TYPES = {**UMLS_DEVICE_TYPES, **UMLS_MAYBE_DEVICE_TYPES}
UMLS_EXTENDED_DIAGNOSTIC_TYPES = {
    **UMLS_DIAGNOSTIC_TYPES,
    **UMLS_MAYBE_DIAGNOSTIC_TYPES,
}
UMLS_EXTENDED_DIAGNOSTIC_TYPES = {
    **UMLS_DIAGNOSTIC_TYPES,
    **UMLS_MAYBE_DIAGNOSTIC_TYPES,
}
UMLS_EXTENDED_MECHANISM_TYPES = {**UMLS_MECHANISM_TYPES, **UMLS_MAYBE_MECHANISM_TYPES}
UMLS_EXTENDED_PROCEDURE_TYPES = {**UMLS_PROCEDURE_TYPES, **UMLS_MAYBE_PROCEDURE_TYPES}
UMLS_UMLS_RESEARCH_TYPES = {**UMLS_RESEARCH_TYPES, **UMLS_MAYBE_RESEARCH_TYPES}

"""
ENTITY_TO_UMLS_TYPE

To see what types are left UNKNOWN:
```
select canonical_id, biomedical_entity.count, biomedical_entity.name, type_ids
from biomedical_entity, umls
where umls.id=canonical_id and entity_type='UNKNOWN' order by count desc limit 1000
```
"""
ENTITY_TO_UMLS_TYPE = {
    BiomedicalEntityType.BIOLOGIC: UMLS_EXTENDED_BIOLOGIC_TYPES,
    BiomedicalEntityType.COMPOUND: UMLS_EXTENDED_COMPOUND_TYPES,
    BiomedicalEntityType.DEVICE: UMLS_EXTENDED_DEVICE_TYPES,
    # includes phenotypes for the sake of typing; phenotype category not ideal for matching since it is very broad
    BiomedicalEntityType.DISEASE: UMLS_EXTENDED_DISEASE_TYPES,
    BiomedicalEntityType.DIAGNOSTIC: UMLS_EXTENDED_DIAGNOSTIC_TYPES,
    BiomedicalEntityType.DOSAGE_FORM: UMLS_FORMULATION_TYPES,
    BiomedicalEntityType.MECHANISM: UMLS_EXTENDED_MECHANISM_TYPES,
    BiomedicalEntityType.PROCEDURE: UMLS_EXTENDED_PROCEDURE_TYPES,
    BiomedicalEntityType.RESEARCH: UMLS_RESEARCH_TYPES,
}

# not used yet
NAME_TO_UMLS_TYPE = {
    "dosage form": BiomedicalEntityType.DOSAGE_FORM,
    "industrial aid": BiomedicalEntityType.INDUSTRIAL,
}

UMLS_TO_ENTITY_TYPE = {v: k for k, vs in ENTITY_TO_UMLS_TYPE.items() for v in vs.keys()}


NER_ENTITY_TYPES = frozenset(
    [
        "biologics",
        "compounds",
        "devices",
        "diseases",
        "mechanisms",
        "procedures",
    ]
)


CANDIDATE_TYPE_WEIGHT_MAP = {
    **{t: 1 for t in list(UMLS_MAYBE_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())},
    **{t: 1 for t in list(UMLS_MAYBE_PHARMACOLOGIC_INTERVENTION_TYPES.keys())},
    **{t: 1 for t in list(UMLS_MAYBE_DISEASE_TYPES.keys())},
    **{t: 1 for t in list(UMLS_PATHOGEN_TYPES.keys())},
    **{t: 1 for t in list(UMLS_NON_PATHOGEN_DISEASE_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_COMPOUND_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_BIOLOGIC_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_MECHANISM_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_TARGET_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_CORE_DISEASE_TYPES.keys())},
    "T200": 0.7,  # Clinical Drug - too specific. avoid matching.
}


PREFERRED_ANCESTOR_TYPE_MAP: dict[str, dict[str, int]] = {
    **{
        k: {
            **{dt: 4 for dt in list(UMLS_MAYBE_DISEASE_TYPES.keys())},
            **{dt: 3 for dt in list(UMLS_PATHOGEN_TYPES.keys())},
            **{dt: 2 for dt in list(UMLS_NON_PATHOGEN_DISEASE_TYPES.keys())},
            **{dt: 1 for dt in list(UMLS_CORE_DISEASE_TYPES.keys())},
        }
        for k in UMLS_EXTENDED_DISEASE_TYPES.keys()
    },
    **{
        k: {
            **{
                dt: 7
                for dt in list(UMLS_MAYBE_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            },
            **{
                dt: 6 for dt in list(UMLS_MAYBE_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            },
            **{dt: 5 for dt in list(UMLS_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())},
            **{dt: 4 for dt in list(UMLS_COMPOUND_TYPES.keys())},
            **{dt: 3 for dt in list(UMLS_BIOLOGIC_TYPES.keys())},
            **{dt: 2 for dt in list(UMLS_MECHANISM_TYPES.keys())},
            **{dt: 1 for dt in list(UMLS_TARGET_TYPES.keys())},
        }
        for k in UMLS_EXTENDED_PHARMACOLOGIC_INTERVENTION_TYPES.keys()
    },
}

PERMITTED_ANCESTOR_TYPES = list(
    set([k for v in PREFERRED_ANCESTOR_TYPE_MAP.values() for k in v.keys()])
)
