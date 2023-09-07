"""
Terms for interventions, used in the biosym_annotations cleanup.
TODO: combine with biologics.py / moa.py / etc.
"""
from utils.re import get_or_re


PRIMARY_BASE_TERMS: dict[str, str] = {
    "activat(?:or|ion|ing|e)?": "activator",
    "agoni(?:st|[sz]ing|[sz]e|sm)?": "agonist",
    "antagoni(?:st|[sz]ing|[sz]e|sm)?": "antagonist",
    "amplif(?:ier|ication|ying|y)": "amplifier",
    "bind(?:er|ing)?": "binder",
    "block(?:er|ing|ade)?": "blocker",
    "chaperone": "chaperone",
    "cataly(?:st|ze|zing)": "catalyst",
    "degrad(?:er|ation|ing|e)?": "degrader",
    "desensitiz(?:ation|ing|e)": "desensitizer",
    "disrupt(?:or|ion|ing)?": "disruptor",
    "desicca(?:tion|nt|te|ted|ting)": "desiccant",
    "enhanc(?:er|ing|e)": "enhancer",
    "induc(?:er|ing|ion|e)?": "inducer",
    "inhibit(?:ion|ing)?": "inhibitor",
    "mimetic": "mimetic",
    "modif(?:ier|ying|ication|y)": "modifier",
    "promot(?:or|ion|ing|ant|e)?": "promoter",
    "potentiat(?:or|ion|ing|e)?": "potentiator",
    "regenerat(?:e|ion|ing|ion)": "regenerator",
    "suppress(?:or|ion|ing|ant)?": "suppressor",
    "stimulat(?:or|ion|ing|ant|e)?": "stimulator",
    "sensitiz(?:ation|ing|e)": "sensitizer",
    "stabiliz(?:er|ing|e|ion)": "stabilizer",
    "transport(?:er|ing|ation)?": "transporter",
}

SECONDARY_MECHANISM_BASE_TERMS = [
    "immun[io][zs](?:ation|ing|e|logical|therapy|genic)",
    "(?:(?:down|up)[ -]?)?regulat(?:or|ion|ing)",
    "(?:(?:neuro|immuno)[- ]?)?modulat(?:or|ion|ing|e)s?(?: binding)?",
    "(?:(?:t[ -]cell )?engag(?:er|ing|e|ment)(?: receptor)?|tce(?:r))",
    "activity",
    "adjuvant",
    "agent",
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
    "(?:(?:neo|peri)[ -]?)?adjuvant",
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

MECHANISM_BASE_TERMS = [
    *list(PRIMARY_BASE_TERMS.keys()),
    *SECONDARY_MECHANISM_BASE_TERMS,
]
BIOLOGIC_BASE_TERMS: list[str] = [
    "(?:(?:glyco|fusion)[ -]?)?proteins?(?: binding)?",
    "(?:poly)peptides?(?: binding)?",
    "(?:poly)?nucleotides?(?: binding)?",
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
    "(?:oligo[ -])?nucleotide",
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
    "agent",
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
    "content",
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
    "human(?:ized|ised)?",
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
    "rodent",
    "selective",
    "single",
    "short",
    "soluble",
    "substituted",
    "such",
    "target(?:ing|ed)?",
    "therapeutic",
    "topical",
    "useful",
    "viable",
]

INTERVENTION_PREFIXES_GENERIC_RE = get_or_re(
    INTERVENTION_PREFIXES_GENERIC,
    "+",
    permit_trailing_space=True,
    enforce_word_boundaries=True,
)

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
    "recobinant",
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
