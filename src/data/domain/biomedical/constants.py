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
    "composition containing": "all",
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
    "therein": "all",
    "thereinof": "all",
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
    "formula [(]?[ivxab]{1,3}[)]?": "all",
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
    # **MOA_PATTERNS,
    # **ACTIVITY_MOA_PATTERNS,
    r"κB": "kappa-b",
    r"nf[- ]?κ[BβΒ]": "nfkb",
    r"(?:α|a|amyloid)[ ]?(?:β|b|beta)[ ]?([-0-9]{1,5})": r"abeta\1",  # scispacy does better with this
    # "tnf α" -> "tnf alpha"
    r"(.*\b)[Αα](\b.*)": r"\1alpha\2",
    r"(.*\b)[βΒ](\b.*)": r"\1beta\2",
    r"(.*\b)[γΓ](\b.*)": r"\1gamma\2",
    r"(.*\b)[δΔ](\b.*)": r"\1delta\2",
    r"(.*\b)[ωΩ](\b.*)": r"\1omega\2",
    r"(.*\b)[ηΗ](\b.*)": r"\1eta\2",
    r"(.*\b)[κ](\b.*)": r"\1kappa\2",
    # "blahα" -> "blaha"
    r"(\w+)[Αα](.*)": r"\1a\2",
    r"(\w+)[βΒ](.*)": r"\1b\2",
    r"(\w+)[γΓ](.*)": r"\1g\2",
    r"(\w+)[δΔ](.*)": r"\1d\2",
    r"(\w+)[ωΩ](.*)": r"\1o\2",
    r"(\w+)[ηΗ](.*)": r"\1e\2",
    r"(\w+)[κ](.*)": r"\1k\2",
    # "γc" -> "gc"
    r"(.*)[Αα](\w+)": r"\1a\2",
    r"(.*)[βΒ](\w+)": r"\1b\2",
    r"(.*)[γΓ](\w+)": r"\1g\2",
    r"(.*)[δΔ](\w+)": r"\1d\2",
    r"(.*)[ωΩ](\w+)": r"\1o\2",
    r"(.*)[ηΗ](\w+)": r"\1e\2",
    r"(.*)[κ](\w+)": r"\1k\2",
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
    "g[-]?pcrs?": "gpcr",
    "non[ -]?steroidal": "nonsteroidal",
    "re[ -]?uptake": "reuptake",
    "toll[ ]?like": "toll-like",
    "t cell": "t-cell",
    "b cell": "b-cell",
    "([a-z]{1,3}) ([0-9]{1,4})": r"\1\2",  # e.g. CCR 5 -> CCR5 (dashes handled in normalize_by_pos)
}
