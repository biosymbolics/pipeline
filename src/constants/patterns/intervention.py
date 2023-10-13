"""
Terms for interventions, used in the biosym_annotations cleanup.
TODO: combine with biologics.py / moa.py / etc.
"""
from utils.re import get_or_re


PRIMARY_BASE_TERMS: dict[str, str] = {
    "activat(?:or|ion|ing|e)?": "activator",
    "(?:super[ -]?)?agoni[sz](?:t|ing|er?|m)?": "agonist",
    "antagoni[sz](?:t|ing|er?|m)?": "antagonist",
    "amplif(?:ier|ication|ying|y)": "amplifier",
    "bind(?:er|ing)?": "binder",
    "block(?:er|ing|ade)?": "blocker",
    "chaperone": "chaperone",
    "(?:co[ -]?|bio[ -]?)?cataly(?:st|ze|zing|zer)": "catalyst",
    "degrad(?:er|ation|ing|e)?": "degrader",
    "desensitiz(?:ation|ing|e|er)": "desensitizer",
    "disrupt(?:or|ion|ing)?": "disruptor",
    "disintegrat(?:or|ion|ing|e)?": "disintegrator",
    "desicca(?:tion|nt|te|ted|ting)": "desiccant",
    "enhanc(?:er|ing|e)": "enhancer",
    "emulsif(?:y|ying|ier)?": "emulsifier",
    "inactivat(?:or|ion|ing|e)?": "inactivator",
    "induc(?:er|ing|ion|e)?": "inducer",
    "inhibit(?:ion|ing|or)?": "inhibitor",
    "initiat(?:er?|ion|ing|or)?": "initiator",
    "introduc(?:er?|ing|tion)": "introducer",
    "mimetic": "mimetic",
    "modulat(?:or|ion|ing|e)?": "modulator",
    "modif(?:ier|ying|ication|y)": "modifier",
    "oxidi[sz](?:er|ing|ation|e)?": "oxidizer",
    "promot(?:or|ion|ing|ant|e)?": "promoter",
    "potentiat(?:or|ion|ing|e)?": "potentiator",
    "regenerat(?:e|ion|ing|ion|or)": "regenerator",
    "suppress(?:or|ion|ing|ant)?": "suppressor",
    "stimulat(?:or|ion|ing|ant|e)?": "stimulator",
    "sensiti[sz](?:ation|ing|e|er)": "sensitizer",
    "strip(?:per|ping|ped|ping)": "stripper",
    "stabili[sz](?:er|ing|e|ion)": "stabilizer",
    "thicken(?:er|ing|ed)": "thickener",
    "transport(?:er|ing|ation)?": "transporter",
}

SECONDARY_MECHANISM_BASE_TERMS = [
    "immun[io][zs](?:ation|ing|e|logical|therapy|genic)",
    "(?:(?:down|up)[ -]?)?regulat(?:or|ion|ing)",
    "(?:(?:neuro|immuno)[- ]?)?modulat(?:or|ion|ing|e)s?(?: binding)?",
    "(?:(?:t[ -]cell )?engag(?:er|ing|e|ment)(?: receptor)?|tce(?:r))",
    "absorb[ae]nt",
    "activity",
    "action",
    "adhesive",
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
    "diluent",
    "disinfectant",
    "dye",
    "effect",
    "emitter",
    "emollient",
    "encoder",
    "expression",  # TODO: also a disease term
    "(?:growth )?factor",
    "interference",
    "interaction",
    "interface",
    "lubricant",
    "metabolite",
    "(?:(?:neo|peri)[ -]?)?adjuvant",
    "phototherap(?:y|ie)",
    "pigment",
    "polymorph",
    "precursor",
    "primer",
    "pro[ -]?drug",
    "pathway",
    "reinforce",  # ??
    "reinforcement",  # ??
    "(?:organic )?solvent",
    "sunscreen",
    "surfactant",
    "target",  # ??
    "transcription(?: factor)?",
    "transfection",
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
    "(?:poly)?nucleo[ts]ides?(?: binding)?",
    "(?:receptor |protein )?ligands?(?: binding| that bind)?",
    "(?:dna |nucleic acid )?fragments?(?: binding| that bind)?",
    "aggregate",
    "amino acid",
    "(?:neo)?antigen",
    "antigen[ -]binding fragment",
    "antigen presenting cell",
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
    "(?:cell )?culture",
    "cytokine",
    "decomposition",
    "double[- ]?stranded rna",
    "epitope",
    "exosome",
    "enzyme",
    "fab(?: region)?",
    "factor [ivx]{1,3}",
    "fc(?:[- ]fusion )?(?: protein)?",
    "fusion" "(?:hairpin|micro|messenger|sh|si|ds|m|hp)?[ -]?rna",
    "(?:trans)?gene",
    "isoform",
    "(?:phospho[ -]?)?lipid",
    "kinase",
    "(?:di|mono|poly|oligo)mer",
    "(?:natural killer|nkt|nkc)(?: cells)?",
    "lignin",
    "mutein",
    "nuclease",
    "(?:oligo[ -])?nucleotide(?: sequence)?",
    "(?:poly[ -]?)?peptide(?: sequence)?",
    "protease",
    "protein",
    "recognizer",
    "scaffold",
    "scfv",  # single-chain variable fragment
    "liposome",
    "(?:expression )?vector",
    "stem cell",
    "(?:tumor[- ]?infiltrating )?lymphocyte",
    "(?:(?:immuno|gene)[ -]?)?therap(?:y|ie)",
]
COMPOUND_BASE_TERMS_SPECIFIC: list[str] = [
    # incomplete is an understatement.
    "acetone",
    "acid",
    "(?:poly)?amide",
    "(?:poly)?amine",
    "ammonia",  # not biomedical?
    "benzene",  # not biomedical?
    "cellulose",
    "(?:immuno[ -]?)?conjugat(?:ion|ing|e)",
    "(?:small |antibody )?molecules?(?: binding)?",
    "carbonyl",
    "diamine",
    "ester",
    "gelatin",
    "glycerin",
    "hydrogel",
    "hydrocarbon",  # not biomedical?
    "indole",
    "(?:stereo)?isomer",
    "ketone",  # not biomedical?
    "phenol",
    "(?:co[ -]?|pre[ -]?)?polymer",
    "propylene",
    "pyridine",
    "resin",  # not biomedical?
    "silicone",
    "starch",
    "sulfur",
    "taxane",  # not biomedical?
    "volatile organic",  # not biomedical?
]


COMPOUND_BASE_TERMS_GENERIC: list[str] = [
    "administration(?: of)?",
    "aerosol",
    "aerogel",
    "agent",
    "applications?(?: of)?",
    "analog(?:ue)?",
    "assembl(?:y|ie)",
    "base",
    "binder",
    "blend",
    "(?:di|nano)?bod(?:y|ie)",  # ??
    "(?:bio)?material",
    "candidate",
    "(?:micro[ -]?|nano[ -]?)?capsule",
    "(?:electro|bio)?[ -]?chemical",
    "combination",
    "complex(?:es)?",
    "composition",
    "component",
    "compound",
    "constituent",
    "content",
    "(?:binding )?(?:compound|molecule)",
    "cream",
    "derivative",
    "detergent",
    "distillation",
    "(?:single )?dose(?:age)?(?: form)?",
    "drop(?:let)",
    "drug",
    "element",
    "enantiomer",
    "expressible",
    "excipient",
    "fluid",
    "foam",
    "form(?:ula|a)?(?:tion)?",
    "function",
    "gel",
    "granule",
    # "group",
    "herb",
    "homolog(?:ue)?",
    "infusion",
    "ingredient",
    "inhalant",
    "injection",
    "laminate",
    "leading",
    "librar(?:y|ies)",
    "(?:producing )?material",
    "medium",
    "member",
    "(?:micro|nano)?[ -]?particle",
    "(?:small|macro|bio)?[ -]?molecule",
    "moiet(?:y|ie)",  # ??
    "motif",  # ??
    "nutrient",
    "nutraceutical",
    "ointment",
    "oil",
    "oral",
    "ortholog(?:ue)?",
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
    "(?:medication |drug |therapy |dose |dosing |dosage |treatment |therapeutic |administration |daily |multiple |delivery |suitable |chronic |suitable |clinical |extended |convenient |effective |detailed |present ){0,3}regimen",
    "reactant",
    "ring",
    "salt",
    "slurr(?:y|ie)",
    "solution",
    "spray",
    "strain",  # ??
    "strand",
    "substance",
    "substitute",
    "substrate",
    "subunit",
    "(?:nutritional )?supplement",
    "support",
    "suppositor(?:y|ie)",
    "tablet",
    "(?:mono[ -]?)?therap(?:y|ies)",
    "therapeutical",
    "treatment",
    "trailing",
    "variant",
    "vehicle",
]

DIAGNOSTIC_BASE_TERMS = [
    "biomarker",
]

COMPOUND_BASE_TERMS = [
    *COMPOUND_BASE_TERMS_SPECIFIC,
    *COMPOUND_BASE_TERMS_GENERIC,
]

OTHER_INTERVENTION_BASE_TERMS = ["sweetener"]


INTERVENTION_BASE_TERMS = [
    *BIOLOGIC_BASE_TERMS,
    *MECHANISM_BASE_TERMS,
    *COMPOUND_BASE_TERMS,
    *DIAGNOSTIC_BASE_TERMS,
    *OTHER_INTERVENTION_BASE_TERMS,
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


INTERVENTION_PREFIXES_GENERIC = [
    "(?:(?:bi|tri|dual|triple)[- ]?)?functional",
    "activ(?:ity|ated)",
    "advanced",
    "asymmetric",  # TODO: move to specific?
    "atypical",
    "capable",
    "contributing",
    "acceptable",
    "(?:biologically )?active",
    "basic",
    "bovine",
    "chemical(?:ly)?(?: modified| formula| structure)*",
    "clinically[ -]?proven",
    "convenient",
    "dual",
    "effective",
    "exemplary",
    "excessive",
    "exceptional",
    "encoding",
    "excellent",
    "function(?:al)?",
    "first",
    "fructose",
    "good",
    "homomeric",  # protein made up of two or more identical polypeptide chains # TODO - specific?
    # "humani[zs]ed",  # TODO: are we sure?
    "inventive",
    "known",
    "mammal(?:ian)?",
    "medical",
    "modified",
    "mouse",
    "murine",
    "muta(?:nt|ted)",
    "native",
    "novel",
    "new",
    "partial",
    "porcine",
    "positive",
    "potent",
    "preferred",
    "prophylactic",
    "promising",
    "proven",
    "pure",
    "optional(?:ly)?",
    "rat",
    "remarkable",
    "rodent",
    "second",
    "single",
    "short",
    "significant",
    "simultaneous(?:ly)?",
    "soluble",
    "striking",
    "substantial(?:ly)?",
    "such",
    "sucrose",
    "super",
    "suitable",
    "target(?:ing|ed)?",
    "therapeutic(?:ally)?",
    "topical",
    "usable",
    "useful",
    "value",
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
    "(?:non-?|mono|spiro|ali|bi|endo|tri|hetero|poly|macro)?cyclic",
    "(?:ion )?channel",
    "inverse",
    "(?:irr)?reversible",
    "negative",
    "receptor",
    "recombinant",
    "selective",
    "(?:un[ -]?)?substituted",
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
