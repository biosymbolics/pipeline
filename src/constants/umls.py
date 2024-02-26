"""
Constants related to UMLS (https://uts.nlm.nih.gov/uts/umls/home)
"""

from typing import Literal, Sequence
from prisma.enums import BiomedicalEntityType

from typings.client import EntityField

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
    "C1505133": "SGLT2",  # otherwise SLC5A2
    "C1550602": "Additive",  # otherwise "Additive (substance)"
    "C1292856": "Stimulator",  # https://uts.nlm.nih.gov/uts/umls/concept/C1292856 Stimulation procedure
    "C0025080": "Device",  # vs "Medical Device"
    "C0348078": "Form",  # qualitative form
    "C0005515": "Biological Factors",
    "C0030281": "Beta Cell",  # Structure of beta Cell of islet
    "C4521924": "CFTR potentiator",  # https://uts.nlm.nih.gov/uts/umls/concept/C4521924
    "C4722064": "Transporter",  # https://uts.nlm.nih.gov/uts/umls/concept/C4722064
    "C1420809": "BCMA",  # TNFRSF17
    "C3203086": "PD-L1 Protein",
    "C0171406": "NADH Dehydrogenase",
    "C0596316": "group",
    "C1254351": "substance",
    "C0013227": "preparation",
    "C0599894": "targeting",
    "C0108801": "tfr1",
    "C0017262": "expression",
    "C3641152": "moiety",
    "C1424685": "nk1",
}

# sets canonical based on (single!) word
UMLS_WORD_OVERRIDES = {
    "modulator": "C0005525",  # "Biological Response Modifiers"
    "modulators": "C0005525",
    "modulation": "C0005525",
    "modulated": "C0005525",
    "modulating": "C0005525",
    "binder": "C1145667",  # binding action
    "binders": "C1145667",  # binding action
    "binding": "C1145667",  # binding action
    "inhibitor": "C1999216",  # https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "inhibitors": "C1999216",  # https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "antagonist": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "antagonists": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "antagonizing": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "antagonism": "C4721408",  # https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "transporter": "C4722064",  # https://uts.nlm.nih.gov/uts/umls/concept/C4722064
    "transporters": "C4722064",  # https://uts.nlm.nih.gov/uts/umls/concept/C4722064
    "ch24h": "C0769789",  # cholesterol 24-hydroxylase
    "agonism": "C2987634",  # agonist
    "sleep initiation and maintenance disorders": "C0851578",
    "hiv/aids": "C0001175",
    "moiety": "C3641152",
    "moieties": "C3641152",
    "tfrl": "C0108801",
    "conjugate": "C4704928",
    "conjugates": "C4704928",
    # "immunoproteasome inhibitor": "C1443643", # Proteasome inhibitor
}

UMLS_CUI_ALIAS_SUPPRESSIONS = {
    "C3470073": ["post"],
    "C0027627": ["secondary"],
    "C1412362": ["flap"],
    "C0029896": ["ent diseases", "disease, ent"],
    "C3540026": ["combinations"],
    "C1420817": ["light"],
    "C0011209": ["delivery"],
    "C1416525": ["alagille syndrome"],
    "C5433528": ["n protein", "protein n"],
    "C1421478": ["wiskott-aldrich syndrome"],
    "C3469826": ["post"],
    "C3812627": ["spasm"],
    "C1419736": ["scar"],
    "C1448132": ["ir"],
    "C1367460": ["warts"],
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
    "C0432616": ["anti a"],
    "C0242781": ["transmission"],  # "disease transmission"
    "C0010124": ["compound b", "b compounds"],  # "corticosterone"
    "C1421222": ["connectin"],  # TTN gene
    "C1418824": ["inhibitor 1", "i1"],  # PPP1R1A
    "C0019167": ["e antigens", "e antigen", "antigen e"],  # "Hepatitis B e Antigens
    "C1448132": ["ir"],
    "C0524983": ["g cell"],
    "C0666364": ["odf"],  # TRANCE protein
    "C0022709": ["ace"],  # Peptidyl-Dipeptidase A
    "C5441529": ["er"],  # Estrogen Receptor Family, human
    "C0072402": ["stk"],  # Protein-Serine-Threonine Kinases # ??
    "C0007578": ["cam"],  # cell-adhesion molecule
    "C0069515": ["neu"],  # erbB-2 Receptor
    "C1422833": ["suppressin"],  # DEAF1
    "C1414085": ["dm"],  # DMPK gene
    "C1413365": ["cf"],  # CFTR gene
    "C1421546": ["x3", "x receptor"],  # XPR1 gene
    "C0015230": ["spots", "rash"],
    "C1420009": ["type", "a4"],  # SGCG gene
    "C0040517": ["ts", "gts"],  # Tourette's syndrome
    "C0004943": ["bd"],  # Behcet Syndrome
    "C0022336": ["cjd"],  # Creutzfeldt-Jakob Syndrome
    "C0175739": ["receptacle", "receptacles"],  # Electrical outlet"
    "C0040188": ["tic", "tics"],  # Tic disorder
    "C0017921": ["amd", "glycogenosis 2"],  # Glycogen storage disease type II
    "C1826790": ["retinal degeneration 3"],  # RD3 gene
    "C0347129": ["ain"],  # anal dysplasia
    "C1422072": ["ct10"],  # MAGEC2 gene
    "C0144227": ["switch 3"],  # SWI3 protein, S cerevisiae
    "C0085113": ["gene, nf 1", "nf 1 gene"],  # NF1 gene
    "C1420817": ["light"],  # TNFSF14 gene
    "C1419349": ["activator 1", "a1"],
    "C0149911": ["hmm", "mahc"],
    "C5575229": ["bind"],
    "C1293131": ["fusion", "fusions"],
    "C0268275": ["ab variant"],
    "C1415363": ["star", "sta receptor"],
    "C1451001": ["star", "sta receptor"],
    "C1448834": ["f5", "ptc"],
    "C0162326": ["sequences", "sequence"],
    "C0171406": ["complex i"],  # NADH Dehydrogenase
    "C0687133": ["interactions"],
    "C2231143": ["multiple uri"],
    "C0031237": ["pertussis"],
    "C1334103": ["allergic rhinitis"],
    "C0085113": [
        "neurofibromatosis",
        "watson disease",
        "on recklinghausen disease",
    ],
    "C1417326": ["b1"],
    "C5779639": ["s. aureus protein a", "staph. protein a"],
    "C0038164": ["protein a", "a protein"],
}

UMLS_COMMON_BASES = {
    "C4704928": "ADC",
    "C2985546": "humanized mab",
    "C0003250": "mab",
    "C0028630": "nucliotide",
    "C0032550": "polynucleotide",
    "C0599013": "Aptamer",
    "C3641152": "moiety",
    "C0017262": "expression",
    "C1335532": "protein family",
    "C1547776": "substance",
    "C1704241": "complex",
    "C2324496": "biogenic peptide",
    "C1999216": "inhibitor",  # matches too much! crowds out useful matches, like targets.
    "C0003241": "antibodies",
    "C0030956": "peptides",
    "C0005525": "modulators",  # Biological Response Modifiers
    "C0243192": "agonists",
    "C4721408": "antagonists",
    "C1305923": "polypeptides",
    "C0023688": "ligands",
    "C1363844": "mediator",
    "C0597357": "receptor",
    "C0003320": "antigens",
    "C0019932": "hormones",
    "C0596973": "monomer",
    "C0014442": "enzymes",
    "C0033684": "proteins",
    "C0017337": "genes",
    "C1517486": "gene expression process",
    "C0007634": "cells",
    "C0001128": "acids",
    "C0002520": "amino acids",
    "C0034788": "receptor, antigen",
    "C1148575": "antigen binding",
    "C0003320": "antigens",
    "C0035668": "rna",
    "C0012854": "dna",
    "C0005515": "biological factors",
    "C0450442": "agent",
    "C5235658": "targeted therapy agent",
    "C0009429": "combo",
    "C1706082": "Compound",
    "C1550600": "ingredient",  # matches too much
    "C0596316": "chemical group",
    "C1963578": "procedure",
    "C0039082": "syndrome",
    "C0012634": "disease",  # useless
    "C1951340": "process",
    "C1254351": "pharmacological Substance",
    "C1550602": "additive",
    "C0013227": "Pharmaceutical Preparations",
    "C0243072": "derivative",  # useless
    "C1167622": "binding",
    "C0008109": "chimera",
    "C0599894": "cell targeting",
}

UMLS_NON_COMPOSITE_SUPPRESSION = {
    **UMLS_COMMON_BASES,
    "C0017362": "genes regulatory",
    "C0017366": "structural genes",
}


UMLS_CUI_SUPPRESSIONS = {
    "C0280950": "cancer related symptom",
    "C0009812": "constitutional symptom",
    "C1457887": "symptom",  # matches too much
    "C0025362": "Mental retardation",
    "C0456981": "specific antigen",
    "C5238790": "medication for kawasaki disease",
    "C0810005": "Congestive heart failure; nonhypertensive",
    "C2697310": "SARCOIDOSIS, SUSCEPTIBILITY TO, 1",
    "C0678661": "biological control",
    "C0178499": "base",
    "C0013216": "pharmacotherapy",
    "C1521826": "protocol agent",
    "C0376315": "manufactured form",
    "C4703569": "urine xenobiotic",  # WTF
    "C1511130": "biochemical process",
    "C1510464": "protein structure",
    "C4277514": "Tripartite Motif Proteins",
    "C3539969": "Specific immunoglobulin combinations",
    "C0003944": "as if personality",  # matches "as if" too much
    "C0017259": "gene conversion",
    "C1298197": "no tumor invasion",
    "C1704222": "genome encoded entity",
    "C0243083": "associated disease",
    "C3263722": "Traumatic AND/OR non-traumatic injury",
    "C0991538": "orderable drug form",
    "C1517360": "gem 220",
    "C0007992": "pharmacological phenomenon",
    "C1698899": "solid drug form",
    "C0525067": "laboratory chemicals",
    "C0963641": "cat combination",
    "C5399721": "Receptor Antagonist [APC]",  # prefer C4721408
    "C1173729": "SPES herbal extract",
    "C0233656": "mental condensation",  # matches condensation
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
    "C0243076": "antagonists",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C4721408
    "C0243192": "agonists",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C2987634
    "C0243077": "inhibitors",  # prefer https://uts.nlm.nih.gov/uts/umls/concept/C1999216
    "C1413336": "cel gene",  # matches cell
    "C0815040": "acidic amino acid",  # matches amino acid
    "C0043335": "xenobiotic",
    "C1257890": "population group",  # matches group
    "C0443933": "Sjogren's syndrome B antibody",  # matches monoclonal antibody (not necessary if semantic matching)
    "C0544791": "Inflammatory fistula",
    "C4085054": "particl",  # matches particle
    "C0596611": "Gene Mutation",
    "C0543419": "Sequela of disorder",
    "C1148560": "molecular_function",
    # "C5235658": "Targeted therapy agent",  # maybe re-enable; it has some utility.
    "C0079429": "gene a",
    "C1260969": "ring device",
    "C0282636": "cell respiration",  # matches respiration
    "C0456386": "Medicament",
    "C0019047": "abnormal hemoglobin",
    "C1516451": "chemical modifier",
    # "C1709058": "modified release dosage form",
    "C2699893": "Molecular Targeted Therapy",  # ??
    "C0004793": "base sequence",  # ??
    "C0728990": "clinical use template",  # template
    "C0085155": "generic drugs",
    "C0336535": "Environmental agent",
    "C0013230": "Investigational new drugs",
    "C0013232": "Drugs, orphan",
    "C0013231": "Drugs, non-prescription",
    "C1517495": "Gene Feature",
    "C0206446": "tin compounds",
    "C0127400": "Mediator brand of benfluorex hydrochloride",  # "mediator" sigh
    "C0439662": "immune",
    "C1514562": "protein domain",
    "C1510464": "protein structure",
    "C0148445": "enhancin",
    "C0037420": "social interaction",
    "C1301751": "no effect",
    "C0221138": "blood group antibody i",
    "C1517495": "gene feature",
}


UMLS_WITH_NO_ANCESTORS = {
    "C0041242": "Trypsin Inhibitors",  # should be C0033607/Protease Inhibitors
    "C4319541": "Apolipoprotein L1",  # should be C4319541/Apolipoproteins L
    "C3850068": "Cytochrome P-450 CYP1A2 Inhibitors",  # should be C3850070/Cytochrome P-450 Enzyme Inhibitors (rel blank)
    "C3494341": "NAV1.8 Voltage-Gated Sodium Channel",  # should be C3494197/Voltage-Gated Sodium Channels (rel blank)
    "C3494197": "Voltage-Gated Sodium Channels",  # should be C0037492/Sodium Channels (rel blank)
    "C1524094": "Recombinant Cytokines",  # should be C0079189/cytokines (but rel doesn't exist; it does however have isa rels)
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
    "and",  # bad for https://uts.nlm.nih.gov/uts/umls/concept/C0021603
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
    "physiology",
    "phenomena",  # e.g. Physiological Phenomena
    "processes",  # e.g. Physiological Processes
    "process",  # too extreme? C2263112
    "pathways",
    "form",
    "wt",  # wt allele
    "consent",
    "reason",  # https://uts.nlm.nih.gov/uts/umls/concept/C0940957
    "unusual",  # C5139132
    "non-neoplastic",  # e.g. non-neoplastic retinal disorder
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

# separate preferred and not preferred to avoid sometimes choosing protein or gene for the same gene symbol.
# important for this to be consistent.
UMLS_PREFERRED_TARGET_TYPES = {
    "T116": "Amino Acid, Peptide, or Protein",
}

UMLS_LESS_PREFERRED_TARGET_TYPES = {
    "T028": "Gene or Genome",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",  # 14666 of 14889 are NOT also in UMLS_PREFERRED_TARGET_TYPES
}

# used for one-off purpose
UMLS_GENE_PROTEIN_TYPES = {
    "T116": "Amino Acid, Peptide, or Protein",
    "T028": "Gene or Genome",
}

UMLS_TARGET_TYPES = {
    **UMLS_PREFERRED_TARGET_TYPES,
    **UMLS_LESS_PREFERRED_TARGET_TYPES,
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
    "T125": "Hormone",  # 2039 of 3348 are NOT also in UMLS_PREFERRED_TARGET_TYPES
    "T126": "Enzyme",  # only 28 or 30924 are NOT also in UMLS_PREFERRED_TARGET_TYPES
    "T192": "Receptor",  # only 198 of 5655 are NOT also in UMLS_PREFERRED_TARGET_TYPES
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
    "biofilm": BiomedicalEntityType.BIOLOGIC,
    "probiotics": BiomedicalEntityType.BIOLOGIC,
    # "agents": irritatingly general
}

UMLS_TO_ENTITY_TYPE = {v: k for k, vs in ENTITY_TO_UMLS_TYPE.items() for v in vs.keys()}


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
    **{t: 1.1 for t in list(UMLS_LESS_PREFERRED_TARGET_TYPES.keys())},
    **{t: 1.2 for t in list(UMLS_PREFERRED_TARGET_TYPES.keys())},
    **{t: 1.1 for t in list(UMLS_CORE_DISEASE_TYPES.keys())},
    "T200": 0.7,  # Clinical Drug - too specific. avoid matching.
}


PREFERRED_ANCESTOR_TYPE_MAP: dict[str, dict[str, int]] = {
    **{
        k: {
            **{dt: 1 for dt in list(UMLS_MAYBE_DISEASE_TYPES.keys())},
            **{dt: 2 for dt in list(UMLS_PATHOGEN_TYPES.keys())},
            **{dt: 3 for dt in list(UMLS_NON_PATHOGEN_DISEASE_TYPES.keys())},
            **{dt: 4 for dt in list(UMLS_CORE_DISEASE_TYPES.keys())},
        }
        for k in UMLS_EXTENDED_DISEASE_TYPES.keys()
    },
    **{
        k: {
            **{dt: 1 for dt in list(UMLS_PATHOGEN_TYPES.keys())},  # probiotics, etc
            **{
                dt: 2
                for dt in list(UMLS_MAYBE_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            },
            **{
                dt: 3 for dt in list(UMLS_MAYBE_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            },
            **{dt: 4 for dt in list(UMLS_NON_PHARMACOLOGIC_INTERVENTION_TYPES.keys())},
            **{dt: 5 for dt in list(UMLS_COMPOUND_TYPES.keys())},
            **{dt: 6 for dt in list(UMLS_BIOLOGIC_TYPES.keys())},
            **{dt: 7 for dt in list(UMLS_MECHANISM_TYPES.keys())},
            **{dt: 8 for dt in list(UMLS_LESS_PREFERRED_TARGET_TYPES.keys())},
            **{dt: 9 for dt in list(UMLS_PREFERRED_TARGET_TYPES.keys())},
        }
        for k in UMLS_EXTENDED_PHARMACOLOGIC_INTERVENTION_TYPES.keys()
    },
}

PERMITTED_ANCESTOR_TYPES = list(
    set([k for v in PREFERRED_ANCESTOR_TYPE_MAP.values() for k in v.keys()])
)


CATEGORY_TO_ENTITY_TYPES: dict[EntityField, Sequence[BiomedicalEntityType]] = {
    "interventions": [
        BiomedicalEntityType.BIOLOGIC,
        BiomedicalEntityType.COMPOUND,
        BiomedicalEntityType.DEVICE,
        BiomedicalEntityType.DOSAGE_FORM,
        BiomedicalEntityType.MECHANISM,
        BiomedicalEntityType.PROCEDURE,
    ],
    "indications": [BiomedicalEntityType.DISEASE],
    "owners": [BiomedicalEntityType.OWNER],
}

ENTITY_TYPE_TO_CATEGORY: dict[BiomedicalEntityType, EntityField] = {
    k: v for v, ks in CATEGORY_TO_ENTITY_TYPES.items() for k in ks
}


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
