"""
Terms for interventions, used in the biosym_annotations cleanup.
TODO: combine with biologics.py / moa.py / etc.
"""
from constants.patterns.biologics import BIOLOGIC_BASE_TERMS
from constants.patterns.iupac import IUPAC_RE, IUPAC_STRINGS
from utils.re import get_or_re


PRIMARY_MECHANISM_BASE_TERMS: dict[str, str] = {
    "activat(?:or|ion|ing|ed?)?": "activator",
    "(?:super[ -]?)?agoni[sz](?:t|ing|er?|m|ed)?": "agonist",
    "antagoni[sz](?:t|ing|er?|m|ed)?": "antagonist",
    "amplif(?:ier|ication|ying|y|ied)": "amplifier",
    "bind(?:er|ing)?": "binder",
    "block(?:er|ing|ade|ed)?": "blocker",
    "chaperone": "chaperone",
    "(?:co[ -]?|bio[ -]?)?cataly(?:st|zed?|zing|zer)(?: system)": "catalyst",
    "degrad(?:er|ation|ing|ed?)?": "degrader",
    "desensitiz(?:ation|ing|ed?|er)": "desensitizer",
    "disrupt(?:or|ion|ing|ed)?": "disruptor",
    "disintegrat(?:or|ion|ing|ed?)?": "disintegrator",
    "desicca(?:tion|nt|te|ting)": "desiccant",
    "engag(?:er|ing|e|ment)": "engager",
    "enhanc(?:er|ing|e)": "enhancer",
    "emulsif(?:y|ying|ier|ied)?": "emulsifier",
    "immunosuppress(?:ion|or|ant)": "immunosuppressant",
    "inactivat(?:or|ion|ing|e)?": "inactivator",
    "induc(?:er|ing|ion|e)?": "inducer",
    "inhibit(?:ion|ing|or|ed)?": "inhibitor",
    "initiat(?:er?|ion|ing|or|ed)?": "initiator",
    "introduc(?:er?|ing|tion|ed)": "introducer",
    "mimetic": "mimetic",
    "modulat(?:or|ion|ing|ed?)?": "modulator",
    "modif(?:ier|ied|ying|ication|y)": "modifier",
    "oxidi[sz](?:er|ed|ing|ation|e)?": "oxidizer",
    "promot(?:or|ion|ing|ant|ed?)?": "promoter",
    "potentiat(?:or|ion|ing|ed?)?": "potentiator",
    "regenerat(?:ed?|ion|ing|ion|or)": "regenerator",
    "suppress(?:or|ion|ing|ant|ed)?": "suppressor",
    "stimulat(?:or|ion|ing|ant|ed?)?": "stimulator",
    "sensiti[sz](?:ation|ing|ed?|er)": "sensitizer",
    "strip(?:per|ping|ped|ping)": "stripper",
    "stabili[sz](?:er|ing|ed?|ion)": "stabilizer",
    "thicken(?:er|ing|ed)": "thickener",
    "transport(?:er|ing|ation|ed)?": "transporter",
}

SECONDARY_MECHANISM_BASE_TERMS = [
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
    "electroluminescent",
    "emitter",
    "emollient",
    "encoder",
    "function",
    "interference",
    "interaction",
    "interface",
    "lubricant",
    "metabolite",
    "(?:(?:neo|peri)[ -]?)?adjuvant",
    "NSAID",
    "phototherap(?:y|ie)",
    "pigment",
    "plasticizer",
    "polymorph",
    "precursor",
    "primer",
    "pro[ -]?drug",
    "pathway",
    "receiver",
    "(?:(?:down|up)[ -]?)?regulat(?:or|ion|ing)",
    "reinforce",  # ??
    "reinforcement",  # ??
    "solvent",
    "sunscreen",
    "surfactant",
    "target",  # ??
    "transfection",
    "transfer",
    "vitamin",
    "vaccine(?: adjuvant)?",
]

MECHANISM_BASE_TERMS = [
    *list(PRIMARY_MECHANISM_BASE_TERMS.keys()),
    *SECONDARY_MECHANISM_BASE_TERMS,
]

COMPOUND_BASE_TERMS_SPECIFIC: list[str] = [
    *IUPAC_STRINGS,
    # incomplete is an understatement.
    "acetone",
    "acid",
    "(?:poly)?amide",
    "(?:poly)?amine",
    "ammonia",  # not biomedical?
    "antimony",
    "benzene",  # not biomedical?
    "cellulose",
    "(?:immuno[ -]?)?conjugat(?:ion|ing|e)",
    "(?:small )?molecules?(?: binding)?",
    "carbonyl",
    "carbene",
    "carbohydrate",
    "diamine",
    "elastomer",
    "emulsion",
    "ester",
    "gelatin",
    "glycerin",
    "heterocyclic",
    "hydrogel",
    "hydrocarbon",  # not biomedical?
    "indole",
    "imidazole",
    "(?:stereo)?isomer",
    "ketone",  # not biomedical?
    "nitrile",
    "nitrate",
    "phenol",
    "(?:co[ -]?|pre[ -]?|bio[ -]?)?polymer",
    "propylene",
    "pyridine",
    "resin",  # not biomedical?
    "silicone",
    "starch",
    "sulfur",
    "triazole",
    "triazine",
    "taxane",  # not biomedical?
    "volatile organic",  # not biomedical?
]


COMPOUND_BASE_TERMS_GENERIC: list[str] = [
    "additive",
    "administration(?: of)?",
    "applications?(?: of)?",
    "analog(?:ue)?",
    "assembl(?:y|ie)",
    "base",
    "binder",
    "blend",
    "by[ -]?product",  # ??
    "(?:di|nano)?bod(?:y|ie)",  # ??
    "(?:lead )?candidate",
    "(?:electro)?[ -]?chemical",
    "(?:re[ -]?)?combination",
    "complex(?:es)?",
    "(?:pharmaceutical |chemical )?composition",
    "component",
    "compound",
    "constituent",
    "content",
    "(?:binding )?(?:compound|molecule)",
    "derivative",
    "detergent",
    "distillation",
    "(?:single )?dose?(?:age|ing)?(?: form| setting)?",
    "drug",
    "element",
    "enantiomer",
    "expressible",
    "excipient",
    "fluid",
    "form(?:ula|a)?(?:tion)?",
    "function",
    # "group",
    "herb",
    "homolog(?:ue)?",
    "ingredient",
    "kit",
    "laminate",
    "leading",
    "librar(?:y|ies)",
    "producing material",
    "medicament",
    "medication",
    "medium",
    "member",
    "(?:micro|nano)?[ -]?particle",
    "(?:small)?[ -]?molecule",
    "modalit(?:y|ie)",
    "moiet(?:y|ie)",  # ??
    "nutrient",
    "nutraceutical",
    "ortholog(?:ue)?",
    "particle",
    "platform",
    "pharmaceutical",
    "preparation",
    "precipitation",
    "product",
    "(?:medication |drug |therapy |dos(?:e|ing|age) |treatment |therapeutic |administration |daily |multiple |delivery |suitable |chronic |suitable |clinical |extended |convenient |effective |detailed |present ){0,3}regimen",
    "reduction",
    "ring",
    "salt",
    "solution",
    "strain",  # ??
    "substance",
    "substitute",
    "substituent",
    "substrate",
    "subunit",
    "(?:nutritional )?supplement",
    "support",
    "(?:mono[ -]?)?therap(?:y|ies)",
    "therapeutical",
    "treatment",
    "trailing",
    "variant",
    "variet(?:y|ie)",
    "vehicle",
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
ALL_COMPOUND_BASE_TERMS_RE = get_or_re(ALL_COMPOUND_BASE_TERMS, "+")
ALL_BIOLOGIC_BASE_TERMS = BIOLOGIC_BASE_TERMS
ALL_MECHANISM_BASE_TERMS = MECHANISM_BASE_TERMS

ALL_BIOLOGIC_BASE_TERMS_RE = get_or_re(ALL_BIOLOGIC_BASE_TERMS, "+")
ALL_MECHANISM_BASE_TERMS_RE = get_or_re(ALL_MECHANISM_BASE_TERMS, "+")
ALL_INTERVENTION_BASE_TERMS = [
    *ALL_COMPOUND_BASE_TERMS,
    *ALL_BIOLOGIC_BASE_TERMS,
    *ALL_MECHANISM_BASE_TERMS,
]
ALL_INTERVENTION_BASE_TERMS_RE = get_or_re(ALL_INTERVENTION_BASE_TERMS)


INTERVENTION_PREFIXES_GENERIC = [
    "(?:(?:bi|tri|dual|triple)[- ]?)?functional",
    "(?:bi|tri|dual|triple|inverse|reverse)(?:[- ]?acting)?",
    "activ(?:ated|atable)",  # not activity
    "advanced",
    "adjunct",
    "asymmetric",  # TODO: move to specific?
    "atypical",
    "bioavailabl",
    "(?:non-?)?aqueous",
    "capable",
    "contributing",
    "acceptable",
    "(?:biologically )?active",
    "basic",
    "bovine",
    "chemical(?:ly)?(?: modified| formula| structure)*",
    "clinically[ -]?proven",
    "convenient",
    "derived",
    "dual",
    "effective",
    "exemplary",
    "excessive",
    "exceptional",
    "encoding",
    "essential",
    "excellent",
    "function(?:al)?",
    "first",
    "fructose",
    "good",
    "homomeric",  # protein made up of two or more identical polypeptide chains # TODO - specific?
    "humani[zs]ed",  # TODO: are we sure?
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
    "(?:in[- ]?)?organic",
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
    "reversible",
    "rodent",
    "second",
    "single",
    "short",
    "significant",
    "simultaneous(?:ly)?",
    "soluble",
    "striking",
    "stringent",
    "substantial(?:ly)?",
    "such",
    "sucrose",
    "super",
    "suitable",
    "target(?:ing|ed)?",
    "(?:chemo[ -]?)?therapeutic(?:ally)?",
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
    "(?:anti[ -]?sense)",
    "aromatic",
    "chimeric",
    "chiral",
    "deuterated",
    "(?:non-?|mono|spiro|ali|bi|endo|tri|hetero|poly|macro)?cyclic",
    "(?:ion )?channel",
    "fused",
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


PROCEDURE_RES = [
    "(?:.* )?ablation",
    "(?:.* )?amplification",
    "(?:.* )?application",
    ".*biopsy",
    "(?:.* )?blow",
    "brachytherapy",
    "(?:.* )?care",
    ".*dialysis.*",
    ".*dissection",
    ".*enema",
    ".*electrical stimulation",
    ".*electroconvulsive.*",
    ".*electrotherapy.*",
    ".*embolization.*",
    "(?:.* )?electrolysis",
    "(?:.* )?extractions?(?: .*)?",
    "(?:.* )?graft(?:ing)?.*",
    ".*graphy",
    "(?:.* )?incision",
    "(?:.* )?imag(?:ed?|ing)(?:.*)?",
    ".*implantat(?:ion|ing).*",
    "(?:.* )?irrigation",
    "(?:.* )?monitor(?:ing)?",
    "(?:.* )?operations?(?: .*)?",
    ".*placement",
    ".*plasty",
    "(?:.* )?procedures?(?: .*)?",
    ".*puncture",  # puncture
    "(?:.* )?radiation",
    ".*radiotherapy",
    "root canal",
    "(?:.* )?extraction",
    ".*scopy",
    "(?:.* )?surger(?:y|ies).*",
    "(?:.* )?technique",
    ".*transplant.*",
    "(?:.* )?ultrasound(?: .*)?",
    ".*wash",
]

BEHAVIOR_RES = [
    ".*behavioral therapy.*",
    ".*counseling.*",
    "(?:.* )?diet",
    ".*exercise.*",
    "(?:.* )?food",
    "(?:.* )?information",
    "(?:.* )?intervention",  # a bold assumption?
    ".*physical therapy.*",
    "(?:.* )?program",
    "(?:.* )?project",
    ".*?psychotherapy",
    ".*rehabilitation.*",
    "(?:.* )?session",
    ".*strateg(?:y|ie)",
    ".*support.*",  # ??
    ".*therapy system",
    "(?:.* )?training",
    "(?:.* )?visits?(?: .*)?",
]


DIAGNOSTIC_RES = [
    ".*biomarker.*",
    "(?:.* )?contrast",
    "(?:.* )?detection",
    "(?:.* )?diagnos(?:tic|is).*",
    ".*examination",
    ".*imaging agent",
    "(?:.* )?screen",
    "(?:.* )?test(?:ing)?",
]

RESEARCH_TOOLS_RES = [
    "(?:.* )?analyte",
    "(?:.* )?assay",
    ".*centrifug.*",
    "(?:.* )?culture",
    "(?:.* )?polymerase chain reaction.*",
    "(?:.* )?media",
    "(?:.* )?microarray",
    "(?:.* )?reagents?(?: .*)?",
    "(?:.* )?reactants?(?: .*)?",
    ".*rna[- ]?seq.*",
    "(?:.* )?sequencing.*",
    "(?:.* )?study",
    ".*western blot.*",
]

PROCESS_RES = [
    "^.+process(?:es|ing|ability)?(?: stream)?",
]

ROA_PREFIX = "(?:epi[ -]?|intra[ -]?|trans[ -]?|sub[ -]?)?"
ROA_VERB = (
    r"(?:solution|administration|route|preparation|application|suspension|composition)?"
)
ROAS = [
    "arterial",
    "articular",
    "buccal",
    "cardiac",
    "cavernous",
    "cerebroventricular",
    "cutaneous",
    "dermal",
    "enteric",
    "enteral",
    "gastrointestinal",
    "(?:inhalation|inhaled)",
    "insufflation",
    "IV",
    "lesion(?:al)?",
    "lingual",
    "labial",
    "muscular",
    "mucosal",
    "nasal",
    "ocular",
    "ophthalmic",
    "oral",
    "osseous",
    "parenteral",
    "peritoneal",
    "rectal",
    "SQ",
    "SUB[-]?Q",
    "thecal",
    "tumor(?:al)?",
    "topical",
    "vaginal",
    "venous",  # intravenous
    "vesical",
    "vitreal",
]

ROA_RE = rf"(?:{ROA_VERB} of )?{ROA_PREFIX}{get_or_re(ROAS)}(?:ly)?[ ]?{ROA_VERB}"

DOSAGE_FORM_RES = [
    "aerosol(?:ized|ization)?",
    "bag",
    "balm",
    "bolus",
    "(?:micro[ -]?|nano[ -]?|soft )?capsule",
    "cream",
    "(?:dose|dosage)",
    "drip",
    "(?:eye |nasal )?drop(?:let)",
    "emulsion",
    "foam",
    "gas(?:eous)?",
    "(?:soft|aero|hydro)?[- ]?gel",
    "granule",
    "infusion",
    "inhal(?:ation|ant)",
    "(?:(?:pressurized )?metered[ -]?dose |p?MDI |dry[ -]powder |DPI )?inhal(?:er|ation|ant)",
    "(?:auto[- ]?)?inject(?:ion|or)",
    "liquid",
    "lotion",
    "lozenge",
    "microsphere",
    "nebuliz(?:er|ation|ant)",
    "ointment",
    "oil",
    "patch",
    "pellet",
    "pill",
    "(?:dry )?powder",
    "ring",
    "shapmoo",
    "slurr(?:y|ie)",
    "spray",
    "syrup",
    "suppositor(?:y|ie)",
    "tablet",
    "tincture",
    "toothpaste",
    "vapor(?:izer)?",
]

DOSAGE_FORM_RE = f"(?:{ROA_RE} )?{get_or_re(DOSAGE_FORM_RES, '+')}"


DOSAGE_UOMS = [
    "mg",
    "milligram",
    "g",
    "gram",
    "mcg",
    "μg",
    "microgram",
    "ml",
    "milliliter",
    "nanogram",
    "ng",
]
