from constants.patterns.intervention import (
    COMPOUND_BASE_TERMS_GENERIC,
    INTERVENTION_PREFIXES_GENERIC,
)


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
    "the": "leading",
    "example": "all",
    "unwanted": "leading",
    "comprised?": "all",
    "contagious": "leading",
    "recognition": "trailing",
    "binding": "trailing",
    "prevention": "leading",
    "that": "trailing",
    "discreet": "all",
    "subject": "leading",
    "properties": "trailing",
    "administration(?: of)?": "all",
    "treatment(?: of)?": "all",
    "library": "all",
    "more": "leading",
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
    "symptom": "trailing",
    "condition": "trailing",
    "be": "trailing",
    "use": "trailing",
    "efficacy": "all",
    "therapeutic procedure": "all",
    "therefor": "all",
    "(?:co[ -]?)?therapy": "trailing",
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
    "medicinal": "all",
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
    "effect": "trailing",
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
EXPAND_CONNECTING_RE = "(?:(?:of|the|that|to|(?:the )?expression|encoding|comprising|with|(?:directed |effective |with efficacy )?against)[ ]?)"
# when expanding annotations, we don't want to make it too long
EXPANSION_NUM_CUTOFF_TOKENS = 7
# leave longer terms alone
POTENTIAL_EXPANSION_MAX_TOKENS = 4

EXPANSION_ENDING_DEPS = ["agent", "nsubj", "nsubjpass", "dobj", "pobj"]
EXPANSION_ENDING_POS = ["NOUN", "PROPN"]

# overrides POS, eg "inhibiting the expression of XYZ"
EXPANSION_POS_OVERRIDE_TERMS = ["expression", "encoding", "coding"]
