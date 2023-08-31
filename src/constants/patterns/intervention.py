"""
Terms for interventions, used in the biosym_annotations cleanup.
TODO: combine with biologics.py / moa.py / etc.
"""
from pydash import flatten
from utils.re import get_or_re


MECHANISM_BASE_TERMS: list[str] = [
    "activity",
    "adjuvant",
    "agent",
    "amplifier",
    "analgesic",
    "anaesthetic",
    "anti[ -]?microbial",
    "anti[ -]?biotic",
    "anti[ -]?infective",
    "anti[ -]?inflammator(?:y|ie)",
    "anti[ -]?bacterial",
    "anti[ -]?viral",
    "anti[ -]?metabolite",
    "carrier",
    "catalyst",
    "chaperone",
    "contraception",
    "co[- ]?factor",
    "(?:cross[ -]?)?linker",
    "decoy",
    "diuretic",
    "donor",
    "differentiation",
    "dilator",
    "disinfectant",
    "effect",
    "emitter",
    "emollient",
    "emulsifier",
    "encoder",
    "factor",
    "growth factor",
    "interference",
    "introducer",
    "interaction",
    "lubricant",
    "metabolite",
    "mimetic",
    "(?:(?:neo|peri)[ -]?)?adjuvant",
    "(?:oligo[ -])?nucleotide",
    "phototherap(?:y|ie)",
    "precursor",
    "primer",
    "pro[ -]?drug",
    "pathway",
    "solvent",
    "surfactant",
    "target",  # ??
    "transcription(?: factor)?",
    "transfer",
    "toxin",
    "vaccine",
    "vaccine adjuvant",
]
BIOLOGIC_BASE_TERMS: list[str] = [
    "antigen",
    "antigen[ -]binding fragment",
    "aptamer",
    "anion",
    "antibod(?:ie|y)",
    "autoantibod(?:y|ie)",
    "adc",
    "antibody[ -]?drug conjugate",
    "bcma(?: nke)?",
    "car[- ]?(?:t|nk)",
    "chimeric antigen receptor.*",
    "chemotherap(?:y|ie)",  # not necessary biologic, more class/descriptive
    "(?:adoptive )?cell(?: tranfer)?(?: therap(?:y|ie))?",
    "clone",
    "cytokine",
    "decomposition",
    "dimer",
    "epitope",
    "exosome",
    "enzyme",
    "fab(?: region)?",
    "factor [ivx]{1-3}",
    "fc(?:[- ]fusion )?(?: protein)?",
    "fusion" "(?:hairpin|micro|messenger|sh|si|ds|m|hp)?[ -]?rna",
    "double[- ]?stranded rna",
    "isoform",
    "(?:phospho[ -]?)?lipid",
    "kinase",
    "(?:mono|poly)mer",
    "(?:natural killer|nkt|nkc)(?: cells)?",
    "neoantigen",
    "nuclease",
    "oligomer",
    "(?:poly[ -]?)?peptide",
    "protease",
    "protein",
    "recognizer",
    "sequence",
    "scaffold",
    "scfv",  # single-chain variable fragment
    "liposome",
    "(?:expression )?vector",
    "stem cell",
    "(?:tumor[- ]?infiltrating )?lymphocyte",
    "(?:(?:immuno|gene)[ -]?)?therap(?:y|ie)",
]
COMPOUND_BASE_TERMS_SPECIFIC: list[str] = [
    "acid",
    "amide",
    "amine",
    "carbonyl",
    "ester",
    "hydrogel",
    "indole",
    "polyamide",
    "polymer",
    "stereoisomer",
]


COMPOUND_BASE_TERMS_GENERIC: list[str] = [
    "administration(?: of)?",
    "aerosol",
    "application",
    "analog(?:ue)?",
    "assembly",
    "base",
    "(?:di|nano)?bodie",  # ??
    "(?:bio)?material",
    "candidate",
    "capsule",
    "chemical",
    "combination",
    "complex(?:e)?",
    "composition",
    "component",
    "(?:binding )?compound",
    "cream",
    "derivative",
    "detergent",
    "dosage form",
    "drop",
    "drug",
    "element",
    "form",
    "formula(?:tion)?",
    "formation",
    "function",
    "gel",
    # "group",
    "herb",
    "homologue",
    "infusion",
    "ingredient",
    "inhalant",
    "injection",
    "leading",
    "medium",
    "(?:micro|nano)?particle",
    "(?:small )?molecule",
    "moiet(?:y|ie)",
    "motif",  # ??
    "nutraceutical",
    "ointment",
    "ortholog",
    "particle",
    "patch",
    "platform",
    "pill",
    "pharmaceutical",
    "powder",
    "preparation",
    "precipitation",
    "product",
    "reagent",
    "regimen",
    "salt",
    "solution",
    "spray",
    "strain",
    "strand",
    "substance",
    "substitute",
    "substrate",
    "subunit",
    "suppository",
    "tablet",
    "therap(?:y|ie)",
    "therapeutical",
    "treatment",
    "trailing",
    "variant",
]

COMPOUND_BASE_TERMS = [
    *COMPOUND_BASE_TERMS_SPECIFIC,
    *COMPOUND_BASE_TERMS_GENERIC,
]

INTERVENTION_BASE_TERMS = [
    *BIOLOGIC_BASE_TERMS,
    *MECHANISM_BASE_TERMS,
    *COMPOUND_BASE_TERMS,
]

MECHANISM_BASE_TERM_SETS: list[list[str]] = [
    [
        "immunization",
        "immunizing",
        "immunize",
        "immunization",
        "immunological",
        "immunoconjugate",
        "immunotherapy",
        "immunogenic composition",
    ],
    ["potentiator", "potentiation", "potentiating", "potentiate"],
    ["regenerate", "regeneration", "regenerating", "regeneration"],
    ["stabilizer", "stabilizing", "stabilize", "stabilization"],
    ["modifier", "modifying", "modification"],
    ["inhibitor", "inhibition", "inhibiting", "inhibit"],
    ["agonist", "agonizing", "agonize", "agonism"],
    ["antagonist", "antagonizing", "antagonize", "antagonism"],
    [
        "activator",
        "activation",
        "activating",
    ],
    [
        "(?:de)?sensitizer",
        "(?:de)?sensitization",
        "(?:de)?sensitizing",
        "(?:de)?sensitize",
    ],
    ["suppressor", "suppression", "suppressing", "suppress", "suppressant"],
    ["stimulator", "stimulation", "stimulating", "stimulate"],
    [
        "(?:(?:neuro|immuno)[- ]?)?modulate",
        "(?:(?:neuro|immuno)[- ]?)?modulates? binding",
        "(?:(?:neuro|immuno)[- ]?)?modulating",
        "(?:(?:neuro|immuno)[- ]?)?modulation",
        "(?:(?:neuro|immuno)[- ]?)?modulator",
    ],
    ["promotion", "promoting", "promote"],
    [
        "(?:t[ -]cell )?engager",
        "(?:t[ -]cell )?engaging(?: receptor)?",
        "(?:t[ -]cell )?engage",
        "(?:t[ -]cell )?engagement",
        # "t[- ]?cell engaging receptor",
        "tce",
        "tcer",
    ],
    ["enhancement", "enhancing", "enhance", "enhancer"],
    [
        "(?:(?:down|up)[ -]?)?regulator",
        "(?:(?:down|up)[ -]?)?regulation",
        "(?:(?:down|up)[ -]?)?regulating",
    ],
    ["degrader", "degradation", "degrading", "degrade"],
    ["inducer", "inducing", "induction"],
    ["promoter", "promoting", "promote", "promotion"],
    ["blocker", "blockade", "blocking", "block"],
    ["transporter", "transporting", "transport", "transportation"],
    ["disruptor", "disruption", "disrupting", "disrupt"],
    [
        "desiccation",
        "desiccant",
        "desiccator",
        "desiccating",
    ],
]

BIOLOGIC_BASE_TERM_SETS: list[list[str]] = [
    [
        "(?:(?:glyco|fusion)[ -]?)?protein",
        "(?:(?:glyco|fusion)[ -]?)?proteins? binding",
    ],
    ["(?:poly)peptide", "(?:poly)peptides? binding"],
    ["(?:poly)?nucleotide", "(?:poly)?nucleotides? binding"],
    [
        "(?:receptor |protein )?ligand",
        "(?:receptor |protein )?ligands? binding",
        "(?:receptor |protein )?ligands? that bind",
    ],
    [
        "(?:dna |nucleic acid )?fragment",
        "(?:dna |nucleic acid )?fragments? binding",
        "(?:dna |nucleic acid )?fragments? that bind",
    ],
]
COMPOUND_BASE_TERM_SETS: list[list[str]] = [
    ["conjugation", "conjugating", "conjugate"],
    ["(?:small |antibody )?molecule", "(?:small |antibody )?molecules? binding"],
]

ALL_COMPOUND_BASE_TERMS = [
    *COMPOUND_BASE_TERMS,
    *flatten(COMPOUND_BASE_TERM_SETS),
]

ALL_COMPOUND_BASE_TERMS_RE = get_or_re([f"{t}s?" for t in ALL_COMPOUND_BASE_TERMS], "+")

ALL_BIOLOGIC_BASE_TERMS = [
    *BIOLOGIC_BASE_TERMS,
    *flatten(BIOLOGIC_BASE_TERM_SETS),
]

ALL_BIOLOGIC_BASE_TERMS_RE = get_or_re([f"{t}s?" for t in ALL_BIOLOGIC_BASE_TERMS], "+")

ALL_MECHANISM_BASE_TERMS = [
    *MECHANISM_BASE_TERMS,
    *flatten(MECHANISM_BASE_TERM_SETS),
]

ALL_MECHANISM_BASE_TERMS_RE = get_or_re(
    [f"{t}s?" for t in ALL_MECHANISM_BASE_TERMS], "+"
)

INTERVENTION_BASE_TERM_SETS: list[list[str]] = [
    *MECHANISM_BASE_TERM_SETS,
    *BIOLOGIC_BASE_TERM_SETS,
    *COMPOUND_BASE_TERM_SETS,
]

ALL_INTERVENTION_BASE_TERMS = [
    *ALL_COMPOUND_BASE_TERMS,
    *ALL_BIOLOGIC_BASE_TERMS,
    *ALL_MECHANISM_BASE_TERMS,
]
ALL_INTERVENTION_BASE_TERMS_RE = get_or_re(ALL_INTERVENTION_BASE_TERMS)


INTERVENTION_BASE_PREFIXES = [
    "acceptable",
    "allosteric",
    "anti",
    "aromatic",
    "(?:(?:bi|mono|multi|poly)[- ]?)?specific",
    "(?:(?:bi|mono|multi|poly)[- ]?)clonal",
    "chimeric",
    "cyclic",
    "dual",
    "encoding",
    "(?:(?:bi|tri|dual|triple)[- ]?)?functional",
    "heterocyclic",
    "inverse",
    "irreversible",
    "(?:ion )?channel",
    "inventive",
    "negative",
    "novel",
    "new",
    "partial",
    "positive",
    "potent",
    "optionally",
    "receptor",
    "reversible",
    "recombinant",
    "topical",
    "tri(?:cyclic)?",
    "selective",
    "single",
    "short",
    "substituted",
    "therapeutic",
    "cd[0-9]{1,2}",
    "cd[0-9]{1,2}-cd[0-9]{2}",
    "cd[0-9]{1,2}xcd[0-9]{2}",  # CD47xCD20
    "cd[0-9]{1,2}x[A-Z]{3,6}",  # CD3xPSCA
    "il[0-9]{1,2}-cd[0-9]{2}",
]
