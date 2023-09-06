"""
Terms for interventions, used in the biosym_annotations cleanup.
TODO: combine with biologics.py / moa.py / etc.
"""
from pydash import flatten
from utils.re import get_or_re


MECHANISM_BASE_TERMS: list[str] = [
    "stabiliz(?:er|ing|e|ion)",
    "modif(?:ier|ying|ication|y",
    "inhibit(?:ion|ing)?",
    "agoni(?:st|[sz]ing|[sz]e|sm)?",
    "antagoni(?:st|[sz]ing|[sz]e|sm)?",
    "activat(?:or|ion|ing|e)?",
    "potentiat(?:or|ion|ing|e)?",
    "suppress(?:or|ion|ing|ant)?",
    "stimulat(?:or|ion|ing|ant|e)?",
    "promot(?:or|ion|ing|ant|e)?",
    "degrad(?:er|ation|ing|e)?",
    "induc(?:er|ing|ion|e)?",
    "block(?:er|ing|ade)?",
    "regenerat(?:e|ion|ing|ion)",
    "(?:de)?sensitiz(?:ation|ing|e)",
    "(?:(?:neuro|immuno)[- ]?)?modulat(?:or|ion|ing|e)s?(?: binding)?",
    "enhanc(?:er|ing|e)" "(?:(?:down|up)[ -]?)?regulat(?:or|ion|ing)",
    "transport(?:er|ing|ation)?"
    "disrupt(?:or|ion|ing)?"
    "desicca(?:tion|nt|te|ted|ting)",
    "immun[io][zs](?:ation|ing|e|logical|therapy|genic)",
    "(?:(?:t[ -]cell )?engag(?:er|ing|e|ment)(?: receptor)?|tce(?:r))",
    "activity",
    "adjuvant",
    "agent",
    "amplifier",
    "analgesic",
    "anaesthetic",
    "anti[ -]?microbial",
    "anti[ -]?biotic",
    "anti[ -]?infect(?:ive|ing)",
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
    "(?:growth )?factor",
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
    "vaccine(?: adjuvant)?",
]
BIOLOGIC_BASE_TERMS: list[str] = [
    "(?:(?:glyco|fusion)[ -]?)?proteins?(?: binding)?",
    "(?:poly)peptides?(?: binding)?" "(?:poly)?nucleotides?(?: binding)?",
    "(?:receptor |protein )?ligands?(?: binding| that bind)?",
    "(?:dna |nucleic acid )?fragments?(?: binding| that bind)?",
    "(?:neo)?antigen",
    "antigen[ -]binding fragment",
    "aptamer",
    "anion",
    "(?:auto)?antibod(?:ie|y)",
    "adc",
    "antibody[ -]?drug[ -]conjugate",
    "bcma(?: nke)?",
    "car[- ]?(?:t|nk)",
    "chimeric antigen receptor",
    "chemotherap(?:y|ie)",  # not necessary biologic, more class/descriptive
    "(?:adoptive )?cell(?: tranfer)?(?: therap(?:y|ie))?",
    "clone",
    "cytokine",
    "decomposition",
    "epitope",
    "exosome",
    "enzyme",
    "fab(?: region)?",
    "factor [ivx]{1,3}",
    "fc(?:[- ]fusion )?(?: protein)?",
    "fusion" "(?:hairpin|micro|messenger|sh|si|ds|m|hp)?[ -]?rna",
    "double[- ]?stranded rna",
    "isoform",
    "(?:phospho[ -]?)?lipid",
    "kinase",
    "(?:di|mono|poly|oligo)mer",
    "(?:natural killer|nkt|nkc)(?: cells)?",
    "mutein",
    "nuclease",
    "(?:poly[ -]?)?peptide",
    "protease",
    "protein",
    "recognizer",
    "scaffold",
    "scfv",  # single-chain variable fragment
    "sequence",
    "liposome",
    "(?:expression )?vector",
    "stem cell",
    "(?:tumor[- ]?infiltrating )?lymphocyte",
    "(?:(?:immuno|gene)[ -]?)?therap(?:y|ie)",
]
COMPOUND_BASE_TERMS_SPECIFIC: list[str] = [
    # incomplete is an understatement.
    "conjugat(?:ion|ing|e)",
    "(?:small |antibody )?molecules?(?: binding)?" "acid",
    "carbonyl",
    "ester",
    "hydrogel",
    "indole",
    "(?:poly)?amide",
    "(?:poly)?amine",
    "(?:stereo)?isomer",
]


COMPOUND_BASE_TERMS_GENERIC: list[str] = [
    "administration(?: of)?",
    "aerosol",
    "application(?: of)?",
    "analog(?:ue)?",
    "assembl(?:y|ies)",
    "base",
    "(?:di|nano)?bodie",  # ??
    "(?:bio)?material",
    "candidate",
    "capsule",
    "chemical",
    "combination",
    "complex(?:es)?",
    "composition",
    "component",
    "(?:binding )?(?:compound|molecule)",
    "cream",
    "derivative",
    "detergent",
    "dosage form",
    "drop",
    "drug",
    "element",
    "form(?:ula|a)?(?:tion)?",
    "function",
    "gel",
    # "group",
    "herb",
    "homolog(?:ue)?",
    "infusion",
    "ingredient",
    "inhalant",
    "injection",
    "leading",
    "medium",
    "(?:micro|nano)?particle",
    "(?:small )?molecule",
    "moiet(?:y|ies)",
    "motif",
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
    "therap(?:y|ies)",
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


ALL_COMPOUND_BASE_TERMS = COMPOUND_BASE_TERMS
ALL_COMPOUND_BASE_TERMS_RE = get_or_re([f"{t}s?" for t in ALL_COMPOUND_BASE_TERMS], "+")
ALL_BIOLOGIC_BASE_TERMS = BIOLOGIC_BASE_TERMS
ALL_MECHANISM_BASE_TERMS = MECHANISM_BASE_TERMS

ALL_BIOLOGIC_BASE_TERMS_RE = get_or_re([f"{t}s?" for t in ALL_BIOLOGIC_BASE_TERMS], "+")


ALL_MECHANISM_BASE_TERMS_RE = get_or_re(
    [f"{t}s?" for t in ALL_MECHANISM_BASE_TERMS], "+"
)

ALL_INTERVENTION_BASE_TERMS = [
    *ALL_COMPOUND_BASE_TERMS,
    *ALL_BIOLOGIC_BASE_TERMS,
    *ALL_MECHANISM_BASE_TERMS,
]
ALL_INTERVENTION_BASE_TERMS_RE = get_or_re(ALL_INTERVENTION_BASE_TERMS)


# TODO split generic and specific; remove generic
INTERVENTION_PREFIXES_GENERIC = [
    "(?:(?:bi|tri|dual|triple)[- ]?)?functional",
    "acceptable",
    "active",
    "basic",
    "bovine",
    "dual",
    "encoding",
    "human(?:ized|ised)",
    "inventive",
    "mammal(?:ian)?",
    "modified",
    "mouse",
    "murine",
    "novel",
    "new",
    "partial",
    "porcine",
    "positive",
    "potent",
    "preferred",
    "optional(?:ly)?",
    "rat",
    "receptor",
    "(?:irr)?reversible",
    "recombinant",
    "rodent",
    "selective",
    "single",
    "short",
    "soluble",
    "substituted",
    "target(?:ing|ed)?" "therapeutic",
    "topical",
    "useful",
]

INTERVENTION_PREFIXES_SPECIFIC = [
    "allosteric",
    "anti",
    "aromatic",
    "chimeric",
    "cyclic",
    "heterocyclic",
    "(?:ion )?channel",
    "inverse",
    "mutant",
    "negative",
    "tri(?:cyclic)?",
    "(?:(?:bi|mono|multi|poly)[- ]?)clonal",
    "(?:(?:bi|mono|multi|poly)[- ]?)specific",
    "cd[0-9]{1,2}",
    "cd[0-9]{1,2}-cd[0-9]{2}",
    "cd[0-9]{1,2}xcd[0-9]{2}",  # CD47xCD20
    "cd[0-9]{1,2}x[A-Z]{3,6}",  # CD3xPSCA
    "il[0-9]{1,2}-cd[0-9]{2}",
]


INTERVENTION_PREFIXES = [
    *INTERVENTION_PREFIXES_GENERIC,
    *INTERVENTION_PREFIXES_SPECIFIC,
]
