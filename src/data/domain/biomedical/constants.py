from constants.patterns.intervention import (
    COMPOUND_BASE_TERMS_GENERIC,
    INTERVENTION_PREFIXES_GENERIC,
    PRIMARY_MECHANISM_BASE_TERMS,
)
from utils.re import get_or_re


from .types import WordPlace


REMOVAL_WORDS_PRE: dict[str, WordPlace] = {
    **{k: "leading" for k in INTERVENTION_PREFIXES_GENERIC},
    "such": "all",
    "method": "all",
    "obtainable": "all",
    "different": "all",
    "-+": "leading",
    "stable": "all",
    "various": "all",
    "responsible": "trailing",
    "the": "leading",
    "associated": "leading",
    "prepared": "leading",
    "example": "all",
    "unwanted": "leading",
    "comprised?": "all",
    "contagious": "leading",
    "compositions that include a": "leading",
    "recognition": "trailing",
    "binding": "trailing",
    "prevention": "leading",
    "that": "trailing",
    "discreet": "all",
    "subject": "leading",
    "properties": "trailing",
    "(?:co[- ]?)?administration(?: of)?": "all",
    "treatment(?: of)?": "all",
    "library": "all",
    "more": "leading",
    "active control": "all",
    "classic": "all",
    "present": "leading",
    "invention": "all",
    "various": "leading",
    "construct": "trailing",
    "particular": "all",
    "uses(?: thereof| of)": "all",
    "designer": "all",
    "obvious": "leading",
    "thereof": "all",
    "specific": "all",
    "in": "leading",
    "more": "leading",
    "a": "leading",
    "non[ -]?toxic": "leading",
    "(?:non )?selective": "leading",
    "adequate": "leading",
    "improv(?:ed|ing)": "all",
    r"\b[(]?e[.]?g[.]?,?": "all",
    "-targeted": "all",
    "long[ -]?acting": "leading",
    "other": "leading",
    "more": "leading",
    "of": "trailing",
    "combined": "trailing",
    "symptom": "trailing",
    "condition": "trailing",
    "be": "trailing",
    "use": "trailing",
    "efficacy": "all",
    "therapeutic procedure": "all",
    "therefor": "all",
    "(?:pre[ -]?)?treatment (?:method|with|of)": "all",
    "treating": "all",
    "contact": "trailing",
    "portion": "trailing",
    "intermediate": "all",
    "suitable": "all",
    "and uses thereof": "all",
    "relevant": "all",
    "patient": "all",
    "thereto": "all",
    "against": "trailing",
    "other": "leading",
    "use of": "leading",
    "certain": "all",
    "working": "leading",
    "on": "trailing",
    "in(?: a)?": "trailing",
    "(?: ,)?and": "trailing",
    "and ": "leading",
    "the": "trailing",
    "with": "trailing",
    "of": "trailing",
    "for": "trailing",
    "=": "trailing",
    "unit(?:[(]s[)])?": "trailing",
    "measur(?:ement|ing)": "all",
    # "system": "trailing", # CNS?
    "[.]": "trailing",
    "analysis": "all",
    "management": "all",
    "accelerated": "all",
    "below": "trailing",
    "diagnosis": "all",
    "fixed": "leading",
    "pharmacological": "all",
    "acquisition": "all",
    "production": "all",
    "level": "trailing",
    "processing(?: of)?": "all",
    "control": "trailing",
    "famil(?:y|ie)": "trailing",
    "(?:pharmaceutically|physiologically) (?:acceptable |active )?": "leading",
    "based": "trailing",
    "an?": "leading",
    "active": "all",
    "wherein": "all",
    "additional": "all",
    "advantageous": "all",
    "aforementioned": "all",
    "aforesaid": "all",
    "efficient": "all",
    "first": "all",
    "second": "all",
    "(?:ab)?normal": "all",
    "inappropriate": "all",
    "formula [(][ivxab]{1,3}[)]": "trailing",
    "is": "leading",
    "engineered": "leading",
    "engineered": "trailing",
    "sufficient": "all",
    "due": "trailing",
    "locate": "all",
    "specification": "all",
    "detect": "all",
    "similar": "all",
    "predictable": "all",
    "conventional": "leading",
    "contemplated": "all",
    "is indicative of": "all",
    "via": "leading",
    "level": "trailing",
    "disclosed": "all",
    "wild type": "all",  # TODO
    "(?:high|low)[ -]?dos(?:e|ing|age)": "all",
    "effects of": "all",
    "soluble": "leading",
    "competitive": "leading",
    # "type": "leading", # type II diabetes
    # model/source
    "murine": "all",
    "monkey": "all",
    "non[ -]?human": "all",
    "primate": "all",
    "mouse": "all",
    "mice": "all",
    "human": "all",  # ??
    "rat": "all",
    "rodent": "all",
    "rabbit": "all",
    "porcine": "all",
    "bovine": "all",
    "equine": "all",
    "mammal(?:ian)?": "all",
}

REMOVAL_WORDS_POST: dict[str, WordPlace] = {
    **dict(
        [
            (t, "conditional_trailing")
            for t in [
                *COMPOUND_BASE_TERMS_GENERIC,
                "activity",
                "agent",
                "effect",
                "pro[ -]?drug",
            ]
        ]
    ),
    **REMOVAL_WORDS_PRE,
    "(?:en)?coding": "trailing",
    "being": "trailing",
    "containing": "trailing",
}


# e.g. '(sstr4) agonists', which NER has a prob with
TARGET_PARENS = r"\([a-z0-9-]{3,}\)"

# no "for", since typically that is "intervention for disease" (but "antagonists for metabotropic glutamate receptors")
# with, as in "with efficacy against"
EXPAND_CONNECTING_RES = [
    "of",
    "the",
    "that",
    "to",
    "(?:the )?expression",
    "encoding",
    "comprising",
    "with",
    "(?:directed |effective |with efficacy )?against",
]
EXPAND_CONNECTING_RE = get_or_re(EXPAND_CONNECTING_RES)
# when expanding annotations, we don't want to make it too long
EXPANSION_NUM_CUTOFF_TOKENS = 7
# leave longer terms alone
POTENTIAL_EXPANSION_MAX_TOKENS = 6

EXPANSION_ENDING_DEPS = ["agent", "nsubj", "nsubjpass", "dobj", "pobj"]
EXPANSION_ENDING_POS = ["NOUN", "PROPN"]

# overrides POS, eg "inhibiting the expression of XYZ"
EXPANSION_POS_OVERRIDE_TERMS = ["directed", "expression", "encoding", "coding"]


# compounds that inhibit ...
MOA_COMPOUND_PREFIX = (
    "(?:compound|composition)s?[ ]?(?:that (?:are|have(?: a)?)|for|of|as|which)?"
)
LOW_INFO_MOA_PREFIX = f"(?:(?:{MOA_COMPOUND_PREFIX}|activity|symmetric|axis|binding|formula|pathway|production|receptor|(?:non )?selective|small molecule|superfamily)[ ])"
GENERIC_COMPOUND_TERM = get_or_re(COMPOUND_BASE_TERMS_GENERIC)

# e.g. "production enhancer" -> "enhancer"
# e.g. "blahblah derivative" -> "blahblah"
MOA_PATTERNS = {
    f"{LOW_INFO_MOA_PREFIX}?{pattern}(?: {GENERIC_COMPOUND_TERM})?": f" {canonical} "  # extra space removed later
    for pattern, canonical in PRIMARY_MECHANISM_BASE_TERMS.items()
}

# inhibitory activity -> inhibitor
ACTIVITY_MOA_PATTERNS = {
    f"{pattern} (?:activity|action|function)": f" {canonical} "
    for pattern, canonical in PRIMARY_MECHANISM_BASE_TERMS.items()
}

# TODO: # 5-нт2а - нт

PHRASE_REWRITES = {
    **MOA_PATTERNS,
    **ACTIVITY_MOA_PATTERNS,
    r"κB": "kappa-b",
    r"nf[- ]?κ[BβΒ]": "nfkb",
    r"(?:α|a|amyloid)[ ]?(?:β|b|beta)[ ]?([-0-9]{1,5})": r"abeta\1",  # scispacy does better with this
    "(.*)[Αα](.*)": r"\1alpha\2",
    "(.*)[βΒ](.*)": r"\1beta\2",
    "(.*)[γΓ](.*)": r"\1gamma\2",
    "(.*)[δΔ](.*)": r"\1delta\2",
    "(.*)[ωΩ](.*)": r"\1omega\2",
    "(.*)[ηΗ](.*)": r"\1eta\2",
    "(.*)[κ](.*)": r"\1kappa\2",
    "analogue": "analog",
    "antibody conjugate": "antibody",
    "antibody immunoconjugate": "antibody",
    "antibodies?": "antibody",
    "antibody(?: construct| drug)": "antibody",
    "associated protein": "protein",
    "associated illness": "associated disease",
    "biologic(?:al)? response modifiers?": "modulator",
    "chimeric[ -]?(?:antigen|antibody)[ -]?receptor": "chimeric antigen receptor",
    "chimeric[ -]?(?:antigen|antibody)[ -]?(?:t[ -]?cell )receptor": "chimeric antigen receptor t-cell",
    "car[ -]t": "chimeric antigen receptor t-cell",
    "conditions and disease": "diseases",
    "disease factors": "diseases",
    "disease states": "diseases",
    "diseases? and condition": "diseases",
    "diseases? and disorder": "diseases",
    "disorders? and disease": "diseases",
    "expression disorders?": "diseases",
    "disease state": "diseases",
    "diseases and condition": "diseases",
    "pathological condition": "diseases",
    "induced diseases": "diseases",
    "mediated condition": "associated disease",
    "mediated disease": "associated disease",
    "related condition": "associated disease",
    "related disease": "associated disease",
    "related illness": "associated disease",
    "induced condition": "associated disease",
    "induced illness": "associated disease",
    "induced disease": "associated disease",
    "induced by": "associated with",
    "family member": "family",
    "family protein": "protein",
    "formulae": "formula",
    # "disease states mediated by": "associated disease",  # disease states mediated by CCR5 (rearrange)
    "diarrhoea": "diarrhea",
    "faecal": "fecal",
    "g[ -]?protein[ -]?(?:coupled|linked)[ -]?receptor": "gpcr",
    "g[-]?pcrs?": "gpcr",
    "gplrs?": "gpcr",
    "homologue": "homolog",
    "ifn": "interferon",
    "immunisation": "immunization",
    "kinases": "kinase",
    "non[ -]?steroidal": "nonsteroidal",
    "protein kinase": "kinase",
    "protein degrader": "degrader",
    "peptide (?:conjugate|sequence|complex(?:es)?)": "peptide",
    "(?:poly)?peptide chain": "polypeptide",
    "polypeptide sequence": "polypeptide",
    "risk of (?:disorder|disease)s?": "diseases",
    "responsive protein": "protein",
    "re[ -]?uptake": "reuptake",
    "(?:therapy|therapeutic) agent": "therapy",
    "target(?:ing)? protein": "protein",
    "target(?:ed|ing) (?:antibody|antibody conjugate)": "antibody",  # TODO - keep ADC? but often abbr as antibody, antibody conjugate, etc.
    "toll[ -]?like": "toll-like",
    "tumour": "tumor",
    "transporter inhibitor": "transport inhibitor",
    "t cell": "t-cell",
    "b cell": "b-cell",
    "interleukin[- ]?([0-9]{1,3})": r"IL\1",
    "il ([0-9]{1,3})": r"IL\1",
    "immunoglobulin ([a-z][0-9]{0,3})": r"IG\1",
    "peginterferon": "pegylated interferon",
    "([a-z]{1,3}) ([0-9]{1,4})": r"\1\2",  # e.g. CCR 5 -> CCR5 (dashes handled in normalize_by_pos)
    "PEG": "pegylated",
    "(?:tgf|transforming growth factor)[ -]?(?:b|β)(?:eta)?(?:[ -]?(?:(?:superfamily )?type )?([0-9]|v?i{1,3}))?": r"tgfb\1",
    # superfamily type ii
    "(?:tgfr?|transforming growth factor(?: receptors?))[ -]?(?:a|α)(?:lpha)?(?:[ -]?([0-9]))?": r"tgfa\1",
    "(?:tnf|tumor necrosis factor)[ -]?(?:a|α)(?:lpha)?(?:[ -]?([0-9]))?": r"tnfa\1",
    "(?:tnf|tumor necrosis factor)[ -]?(?:b|β)(?:beta)?(?:[ -]?([0-9]))?": r"tnfb\1",
    "(?:egf|epidermal growth factor)": r"egf",
    "(?:egfr|epidermal growth factor receptor)(?:[ ]?(v?i{1,3}|[0-9]))?": r"egfr \1",
    # # vascular endothelial growth factor (VEGF), VEGFR1
    # # fibroblast growth factor (FGF), fibroblast growth factor receptor 2
}


DELETION_TERMS = [
    "[(][0-9a-z]{1,4}[)]?[.,]?[ ]?",
    "[0-9., ]+",  # del if only numbers . and ,
    # mangled
    "having formula",
    "acid addition salt",
    "potentially useful",
    "above protein",
    "operating same",
    "device capable",
    "compounds for the",
    "pre[ -]?treatment",
    "enhancing the",
    "non-overlapping double-stranded regions separated",  # ??
    "(?:.* )? capable",  # material capable, etc
    "salt as an",
    "further",
    "individual suffering",
    ".{1,5}-",  # tri-
    ".*(?:composition|compound|substance|agent|kit|group)s? (?:useful|capable)",
    "optionally other modification",
    "co[- ]?operate",
    "light[ -]?receiving",
    "resultant",
    "optionally other",
    "above[ -]?(?: mentioned)?",
    ".* of",
    ".* comprising",
    ".* tubular",
    "composition (?:contain|compris)",
    "[a-zA-Z0-9]+-containing",
    "by-",
    "cur",
    "co-",
    "therefrom",
    "said .*",
    "quantify",
    "citrat",
    "functional",
    "cross[ -]?linking",
    "biocompatibility",
    "particle diamet",
    "embodiment",
    "quantificat",
    "(?:.* )?rate",
    "correspond",
    "water[- ]?soluble",
    "illustrat",
    "cross[- ]?link",
    "antigen-binding",
    "sterilization",
    "regeneration",
    "particulat",
    "embodiment",
    "least partial",
    "reform",
    "conformat(?:ion|ional)",
    # thing (unwanted because generic)
    "material",
    "compositions comprising the compounds",
    "support arm",
    "information system",
    "support structure",
    "support system",
    "backing material",
    "operation part",
    "oxide",
    "agricultural",
    "system",
    "receptor",
    "product .*",
    "pathogen(?:ic)?",
    "regenerative medicine",
    "basic amino acid",
    "(?:.* )?characteristic",
    "response",
    "single piece",
    "product gas",
    r"agent\(s\)",
    "medical purpose",
    "cell membrane",
    "(?:product|reaction) (?:mixture|product|solution|vessel|mixture|system|medium)",
    "(?:.* )?activit",
    "(?:.* )?member",
    "group",
    "product stream",
    "operator",
    "field of(?:.* )",
    # thing (unwanted because wrong type of thing)
    "(?:.* )?propert(?:y|ie)",
    "constraint",
    "(?:side|adverse)[ -]?effect",
    "leaflet",
    "passageway",
    "ability",
    "determinat",  # determination
    "anatomical structure",
    "distract",  # distraction
    "(?:.* )?configuration",
    "considerat",  # consideration
    "(?:.* )?arrangement",
    "(?:.* )?position",
    "(?:.* )?frame",
    "(?:.* )?feature",
    "heat generat",
    "(?:.* )?industry",
    "impact modifier",
    "scratch resistance",
    "gene transfer",
    "mother milk",
    "drug development",
    "(?:.* )?patient",
    "(?:.* )?pest",
    "(?:.* )?examinee",
    "(?:.* )?staff",
    "(?:.* )?trial",
    "(?:.* )?infant",
    "(?:.* )?prospect",
    "(?:.* )?room",
    "professional",
    "(?:.* )?person(?:nel)?",
    "guest",
    "body part",
    "(?:.* )?patent",
    "(?:.* )?pathway",
    "(?:.* )?animal",
    "(?:.* )?retardant",  # e.g. flame retardant
    "aroma",
    "(?:.* )?cosmetic.*",  # may re-enable in the future
    "(?:.* )?hair condition",  # may re-enable in the future
    "(?:.* )?fragrance",
    "silica",
    "perfum",
    "bacteri(?:um|a)",
    "(?:.*)?station",
    "(?:.* )?location",
    "(?:.* )?mode",
    "(?:.* )?region",
    "(?:tumou?r|eukaryotic|liv(?:e|ing)|normal|animal|bacterial|yeast|single|skin|cancer(?:ous)?|insect|host|biological|isolated|primary|diseased?|plant|cultur(?:ing|ed?)|individual) cell",
    "titanium dioxide",
    "microorganism",
    "(?:.* )?area",
    "(?:.* )?power",
    "(?:.* )?site",
    "(?:.* )?signal",
    "(?:.* )?layer",
    "(?:.* )?surface",  # device?
    # effect
    "cell growth",
    "clean",  # cleaning
    "friction",
    "compressive force",
    "correcting dysfunction",
    # "bleaching",
    "vibrat",
    "induc(?:es?d?|ing) differentiation",
    "cool",
    "connect",
    "deterioration",
    "detrimental",
    "(?:.* )?disengage",
    "adjustment",
    "(?:.* )?abrasive",
    "(?:.* )?retardancy",  # e.g. flame retardant
    "film[ -]?form",
    "heat transfer",
    "cell death",
    "deformable",
    "(?:.* )?addition",
    "ameliorat",
    "transformant",
    "deformation",
    "reduction",
    "fixation",
    "stiffen",
    "suction",
    # disease
    "disease state",
    "dysfunction",
    "related (?:disorder|condition|disease)",
    "(?:disorder|condition|disease)s related",
    "disease",
    "syndrome",
    "disorder",
    "dysfunction",
    # part or characteristic of body, or fluid
    "rib",
    "back",
    "cardiovascular",
    "(?:.* )?urine",
    "(?:.* )?appendage",
    "(?:.* )?ventricle",
    "(?:.* )?aorta",
    "(?:.* )?nostril",
    "(?:.* )?nose",
    "(?:.* )?intervertebral disc",
    "(?:.* )?mucosa",
    "(?:.* )?retina",
    "(?:.* )?artery",
    "(?:.* )?vein",
    "(?:.* )?tissue",
    "(?:.* )?septum",
    "(?:.* )?vasculature",
    "(?:.* )?tendon",
    "(?:.* )?ligament",
    "bone fragment",
    "vertebral",
    "(?:.* )?obturator",
    "(?:.* )?atrium",
    "tibial?",
    "(?:.* )?femur",
    "(?:.* )?core",  # device
    "(?:.* )?vesicle",
    "(?:.* )?valve",  # device
    "(?:.* )?atrium",
    "(?:.* )?nail",  # device
    "(?:.* )?joint",
    "(?:.* )?cavity",
    "skin",
    "hair",
    "vascular",
    "capillary",
    "bodily",
    "cornea",
    "vertebra",
    "spine",
    "eye",
    "urea",
    "blood",
    "gastric",
    "gastrointestinal(?: tract)?",
    "cartilage",
    "jaw",
    "liver",
    "heart",
    "ankle",
    "intraocular len",
    "femoral",
    "respiratory(?: tract)?",
    "pulmonary",
    "uterus",
    "lung",
    "plasma",
    "spinal column",
    "muscle",
    "kidney",
    "prostate",
    "pancreas",
    "ocular",
    "spleen",
    "gallbladder",
    "bladder",
    "ureter",
    "urethra",
    "esophagus",
    "stomach",
    "intestin(?:e|al)",
    "colon",
    "rectum",
    "trachea",
    "bronch(?:ial|us)",
    "larynx",
    "pharynx",
    "nasal",
    "sinus",
    "arterial",
    "venous",
    "lymph(?:atic)?(?: node| vessel)?",
    # control
    "(?:.* )?placebos?(?: .*)?",
    "standard treatment",
    "standard of care",
    # category errors
    "buttress",
    "wastewater",
    "individual",
    "formability",
    "(?:fe)?male",
    "(?:.*)?dimensional",
    "pathological",
    "consideration",
    "combin",
    "functionaliz",
    "plurality",
    "physical",
    "demonstrat",
    "engaged position",
    "cell[- ]?free",
    "contribut",
    "advantage",
    "(?:.* )?side",  # base end side
    "accept",
    "solid state",
    "susceptible",
    "(?:.* )?typ",  # type
    "processability",
    "auxiliary",
    "present tricyclic",
    "wherein(?: a)?",
    "particle size",
    "fragment of",
    "concentric(?:al)?",
    "(?:.* )?coupled",
    "(?:.* )?axis",
    "(?:.* )?path",
    "(?:.* )?pattern",
    "(?:.* )?plan",
    "(?:.* )?stage",  # ??
    "(?:.* )?quantit",
    "functional group",
    "drug discovery",
    "combinatorial",
    "compound having",
    "non-",
    # generic
    "mechanical",
    "compound of general",
    "compositions of matter",
    "compounds of the formula",
    "(?:.* )?period",  # never used for menstruation
    "integrated system",
    "renewable resource",
    "pillar",
    "conditioned medium",
    "potential difference",
    "addition salt",
    "layered product",
    "indicia",
    "addition salt",
    "salt of the compound",  # TODO: possible noun exception
    "starting material",
    "multi[- ]?component",
    "discret",  # discrete
    "problem",
    "further",
    "parent",
    "structure(?: directing)?",
    "technology",
    "branch",
    "(?:leading )?edge",
    "approach",
    "extension",
    "space",
    "point",
    "mount",
    "wall",
    "cell(?: cycle| wall| line)",
    "channel",  # todo?
    "design",
    "general structural",
    "piece",
    "attribute",
    "preform",
    "(?:medical|treatment|operation|reaction) (?:fluid|zone|container|section|technology)",
    "object",
    "(?:drug )?target",
    "biologically active",
    "(?:.* )?sample",
    "therapeutically",
    "pharmaceutically-",
    "physiological",
    "pharmaceutically[- ]accept",
    "therapy",
    "(?:general |chemical )?formula(?: i{1,3}){0,3}(?: [0-9a-z]){0,3}",
    "components?(?: i{1,3})?(?: [0-9a-z])?",
    "compounds?(?: i{1,3})?(?: [0-9a-z])?",
    "(?:medicinal|bioactive|biological)(?: activity)?",
    "(?:pharmaceutical|pharmacological|medicinal) composition(?: containing| comprising)?",
    "medicine compound",
    "wherein said compound",
    "pharmaceutical",
    "pharmacological",
    "(?:.* )?compounds? of(?: formula)?",
    "compound of",
    "(?:pharmacologically|pharmaceutically|biologically) active agent",
    "(?:bioactive |medicinal )?agent",
    "chemical entit",
    # general characteristic
    "sanitary",
    "instant",
    "purity",
    "hydrophobic",
    "serum-free",
    "moldability",
    "cellulosic",
    "transgenic",
    "utility",
    "detachabl",
    "proximal",
    "hydrogenat",  # hydrogenated
    "chiral",
    "embolic",
    "traction",
    "annular",
    "molten",
    "(?:solid )?phase",
    "disinfect",  # disinfectant is fine, but not disinfecting
    "(?:open |closed )?configuration",
    "dispersib(?:le|ilit)",
    "photodynamic",
    "structural",
    "accurate",
    "bioavailabilit",
    "usefulness",
    "(?:.* )?conductivit",
    "multi[ -]?function",
    "elastic(?:ity)?",
    "symmetric(?:al)?",
    "biocompatible",
    "biocompatibilt",
    "bioactivit",
    "medicinal",
    "cellular",
    "natural",
    "substantially free",
    "therapeutically active",
    # characteristics / descriptors
    "minimally invasive",
    "adhesiveness",
    "helical",
    "ingestibl",
    "humaniz",  # humanized
    "sheet-like",
    "hydrophobicity",
    "(?:high |low )?concentration",
    "(?:.* )?acceptable",  # e.g. agrochemically acceptable
    "solubilit",
    "(?:.* )?refractive index",
    "uniform(?:it)?",
    "(?:.* )?conductive",
    "granular",
    "luminescent",
    "(?:.* )?bound",
    "amorphous",
    "purif",  # purified
    "hardness",
    "deactivate",  # ??
    "(?:.* )?stability",
    "planar",
    "fatty",
    "compressible",
    "(?:.* )?bond",
    "(?:.* )?perpendicular",
    "(?:.* )?specificity",
    "(?:.* )?domain",
    "storage stability",
    "conductive",
    "flexible",
    "dermatological",
    "bifunctional",
    "in vitro",
    "fibrous",
    "biodegrad",
    "resilient",
    "fluorescent",
    "superabsorbent",
    "non[- ]?woven(.*)?",
    "crystalline",
    "volatile",
    "phenolic",
    "edibl",
    "(?:non[ -]?|potentially )?therapeutic",
    "water[ -]?insoluble",
    "unsaturat",
    "adhes",
    "porous",
    "dispens",
    "impedanc",
    "radioact",
    "optionally substitut",
    "non[ -]?invasive",
    "reinforc",
    "aqueous",
    "concentration",
    "(?:.* )?acidic",
    "(?:.* )?unsubstituted",
    "(?:.* )?synergistic",
    "(?:.* )?hydrophilic",
    "(?:.* )?biodegradabilit",
    "(?:.* )?affinity",
    "(?:.* )?residual",
    "(?:.* )?rigid",
    "(?:.* )?adverse",
    # physical process
    "(?:.* )?reaction",
    "elut",  # elution
    "drug release",
    "sustained[ -]?release",
    "disintegrat",
    "evaporat",
    "agglomerat",
    # measurable thing
    "(?:.* )?movement",
    "(?:.* )?measurement",
    "(?:.* )?curvature",
    "(?:.* )?degree",
    "spatial resolution",
    "oxygen concentrat",
    "(?:.* )?number",
    "(?:.* )?cost",
    "(?:.* )?distance",
    "(?:.* )?frequency",  # also procedure or device
    "(?:.* )?torque",
    "(?:.* )?divergence",
    "(?:.* )?weight",
    "(?:.* )?wavelength(?: .*)?",
    "(?:.* )?charge",
    "(?:.* )?band",
    "(?:.* )?viscosity",
    "(?:.* )?volume",
    "(?:.* )?distribution",
    "(?:.* )?amount",
    "refractive index",
    "phase change",
    "(?:.* )?pressure",
    "(?:.* )?strength",
    "(?:.* )?population",
    "(?:.* )?end",
    "(?:.*[ -])?temperatures?(?: range|high(?:er)?|low(?:er)?)?",
    "(?:.* )?ratio",
    "(?:.* )?deliver",
    "(?:.* )?step",
    "(?:.* )?value",
    "(?:.* )?time",
    # intangible thing
    "(?:.* )?concept",
    "(?:.* )?data",
    "(?:.* )?integer",
    "(?:.* )?language",
    "(?:.* )?information",
    "(?:.* )?report",
    "(?:.* )?parameter",
    "(?:.* )?constant",
    "(?:.* )?longitudinal",
    "(?:.* )?equivalent",
    "(?:.* )?subset",
    "(?:.* )?memory",
    # food
    "(?:.* )?fermentation",
    # material or materials
    "(?:.* )?metal",
    "cobalt",
    "block copolymer",
    "chromium",
    "polymerizable",
    "(?:.* )?metal oxide",
    "(?:.* )?graphite",
    "(?:.* )?graphene(?: oxide)?",
    "(?:.* )?metal",  # temporary?
    # non-medical procedure or process"
    "solid-liquid separation",
    "mix",
    "pattern form",  # pattern forming
    "crystallizat",
    "quantitat",
    "(?:.* )?communicat",
    "(?:.* )?(?:micro)?process",
    # agtech
    "feed(?: material|substance)?",
    "pest control",
    "(?:.* )?feedstock",
    "(?:.* )?biomass",
    "(?:.* )?agrochemical",
    "(?:.* )?herbicid(?:e|al)(?: .*)?",
    "(?:.* )?insecticid(?:e|al)(?: .*)?",
    # "(?:.* )?biocid(?:e|al)(?: .*)?", # could be biomedical
    # "(?:.* )?fungicid(?:e|al)(?: .*)?",  # could be biomedical
    "(?:.* )?pesticid(?:e|al)(?: .*)?",
    "(?:.* )?plant(?: cell)?",
    "drought tolerance",
    "biofuel",
    "biodiesel",
    "plant growth regulator",
    # end agtech
    # start industrial
    "coating composition",
    "carboxylic acid",  # maybe keep?
    "heat resistance",
    "methanol",
    "methane",
    "phosphate",
    "thermoset",  # thermosetting
    "solvent system",
    "biodegradable polymer",
    "corros",  # corrosion
    "asphalt",
    "heat exchang",
    "carbon black",
    "(?:.* )?composite",
    "(?:.* )?manufactur",
    "propane",
    "(?:.* )?energy",
    "electric(?:al)(?:ly)?.*",
    "formaldehyde",  # ??
    "metal catalyst",
    "styrene",
    "curing agent",
    "carbon dioxide",
    "isocyanate",
    "hydrogen",  # industrial
    "aromatic ring",  # industrial
    "polymer matrix",  # industrial
    "(?:quaternary )?ammonium",
    "polyolefin",  # industrial
    "polyisocyanate",  # industrial
    "alkaline",  # industrial
    "trifluoromethyl",  # industrial
    "thermoplastic(?: .*)?",  # industrial
    "plastic material",
    "(?:.* )?resin",
    "(?:.* )?epoxy",
    "(?:.* )?polyurethane",
    "ethylene",
    "alkylat",
    "carbonyl",
    "aldehyde",
    # end industrial
    # start military
    "explosive",
    # end military
]