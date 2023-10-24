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


LOW_INFO_MOA_PREFIX = "(?:(?:asymmetric|axis|binding|formula|pathway|production|receptor|(?:non )?selective|small molecule|superfamily)[ ])"
GENERIC_COMPOUND_TERM = get_or_re(COMPOUND_BASE_TERMS_GENERIC)

# e.g. "production enhancer" -> "enhancer"
# e.g. "blahblah derivative" -> "blahblah"
MOA_PATTERNS = {
    f"{LOW_INFO_MOA_PREFIX}?{pattern}(?:[ ]{GENERIC_COMPOUND_TERM})?": f" {canonical} "  # extra space removed later
    for pattern, canonical in PRIMARY_BASE_TERMS.items()
}

# inhibitory activity -> inhibitor
ACTIVITY_MOA_PATTERNS = {
    f"{pattern}(?:[ ]activity|action)": f" {canonical} "
    for pattern, canonical in PRIMARY_BASE_TERMS.items()
}


PHRASE_MAP = {
    **MOA_PATTERNS,
    **ACTIVITY_MOA_PATTERNS,
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
    # "disease states mediated by": "associated disease", # disease states mediated by CCR5 (rearrange)
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
    "(?:tgf|transforming growth factor)[ -]?(?:b|β)(?:eta)?(?:[ -]?(?:(?:superfamily )?type )?([0-9]|v?i{1,3}))?": r"tgfβ\1",
    # superfamily type ii
    "(?:tgf|transforming growth factor)[ -]?(?:a|α)(?:lpha)?(?:[ -]?([0-9]))?": r"tgfα\1",
    "(?:tnf|tumor necrosis factor)[ -]?(?:a|α)(?:lpha)?(?:[ -]?([0-9]))?": r"tnfα\1",
    "(?:tnf|tumor necrosis factor)[ -]?(?:b|β)(?:beta)?(?:[ -]?([0-9]))?": r"tnfβ\1",
    "(?:tnf|tumor necrosis factor)[ -]?(?:g|γ)(?:amma)?(?:[ -]?([0-9]))?": r"tnfγ\1",
    "(?:tnfr|tumor necrosis factor receptors?)[ -]?(?:a|α)(?:lpha)?(?:[ -]?([0-9]))?": r"tnfrα\1",
    "(?:egf|epidermal growth factor)": r"egf",
    "(?:egfr|epidermal growth factor receptor)(?:[ ]?([v?i{1,3}|[0-9]))": r"egfr\1",
    # vascular endothelial growth factor (VEGF), VEGFR1
    # fibroblast growth factor (FGF), fibroblast growth factor receptor 2
}
