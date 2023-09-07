from constants.patterns.intervention import (
    COMPOUND_BASE_TERMS_GENERIC,
    PRIMARY_BASE_TERMS,
)
from utils.re import get_or_re


NER_TYPES = [
    {
        "dataset": "BIOSYM",
        "name": "compounds",
        "description": "in this context, compounds are chemical or biological substances, drug classes or drug names",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "diseases",
        "description": "diseases are conditions and indications for which pharmacological treatments are developed",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "mechanisms",
        "description": "mechanisms of action are ways in which a drug has an effect (inhibitors, agonists, etc)",
        "description_source": "",
    },
]


LOW_VALUE_MOA_PREFIX = (
    "(?:(?:axis|binding|formula|pathway|receptor|(?:non )?selective|small molecule)[ ])"
)

LOW_VALUE_MOA_POSTFIXES = [
    *COMPOUND_BASE_TERMS_GENERIC,
    "activ(?:ity|ation|ed)",
    "actions?",
    "capable",
    "contributing",
    "effects?",
    "functions?",
    "ligands?",
    "pathways?",
    "(?:poly)?peptides?",
    "proteins?",
    "useful",
]
LOW_VALUE_MOA_POSTFIX = get_or_re(
    LOW_VALUE_MOA_POSTFIXES,
    "+",
    permit_trailing_space=True,
    enforce_word_boundaries=True,
)


MOA_PATTERNS = {
    f"{LOW_VALUE_MOA_PREFIX}?{pattern}[ ]{LOW_VALUE_MOA_POSTFIX}?": f" {canonical} "  # extra space removed later
    for pattern, canonical in PRIMARY_BASE_TERMS.items()
}

PHRASE_MAP = {
    **MOA_PATTERNS,
    "analogue": "analog",
    "activating": "activator",
    "antibody conjugate": "antibody",
    "antibody immunoconjugate": "antibody",
    "antibodies?": "antibody",
    "antibody(?: construct| drug)": "antibody",
    "associated protein": "protein",
    "associated illness": "associated disease",
    "biologic(?:al)? response modifiers?": "immunomodulator",
    "chimeric[ -]?(?:antigen|antibody)[ -]?receptor": "chimeric antigen receptor",
    "chimeric[ -]?(?:antigen|antibody)[ -]?(?:t[ -]?cell ) receptor": "chimeric antigen receptor t-cell",
    "car[ -]t": "chimeric antigen receptor t-cell",
    "conditions and disease": "diseases",
    "disease factors": "diseases",
    "diseases and disorder": "diseases",
    "disorders and disease": "diseases",
    "disease state": "diseases",
    "diseases and condition": "diseases",
    "induced diseases": "diseases",
    "mediated condition": "associated disease",
    "mediated disease": "associated disease",
    "related condition": "associated disease",
    "related disease": "associated disease",
    "related illness": "associated disease",
    "induced condition": "associated disease",
    "induced illness": "associated disease",
    "induced disease": "associated disease",
    "family member": "family",
    "family protein": "protein",
    # "disease states mediated by": "associated disease", # disease states mediated by CCR5 (rearrange)
    "diarrhoea": "diarrhea",
    "faecal": "fecal",
    "g[ -]?protein[ -]?coupled[ -]?receptor": "g protein-coupled receptors",
    "gpcrs?": "g protein-coupled receptors",
    "homologue": "homolog",
    "ifn": "interferon",
    "kinases": "kinase",
    "non[ -]?steroidal": "nonsteroidal",
    "protein kinase": "kinase",
    "protein degrader": "degrader",
    "peptide complex(?:es)?": "peptide",
    "(?:poly)?peptide chain": "polypeptide",
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
    "interleukin[- ]?([0-9]+)": r"IL\1",
    "il ([0-9]+)": r"IL\1",
    "immunoglobulin ([a-z][0-9]*)": r"IG\1",
    "peginterferon": "pegylated interferon",
    "([a-z]{1,3}) ([0-9]+)": r"\1\2",  # e.g. CCR 5 -> CCR5 (dashes handled in normalize_by_pos)
    "PEG": "pegylated",
}
