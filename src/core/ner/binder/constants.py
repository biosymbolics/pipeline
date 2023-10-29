from constants.patterns.intervention import (
    COMPOUND_BASE_TERMS_GENERIC,
    PRIMARY_BASE_TERMS,
)
from utils.re import get_or_re


NER_TYPES = [
    {
        "dataset": "BIOSYM",
        "name": "compounds",
        "description": "any chemical substance that when consumed causes a change in the physiology of an organism, including its psychology. Drugs are typically distinguished from food and other substances that provide nutritional support.",
        "description_source": "https://en.wikipedia.org/wiki/Drug",
    },
    {
        "dataset": "BIOSYM",
        "name": "diseases",
        "description": "A disease is a particular abnormal condition that adversely affects the structure or function of all or part of an organism and is not immediately due to any external injury.",
        "description_source": "https://en.wikipedia.org/wiki/Disease",
    },
    {
        "dataset": "BIOSYM",
        "name": "mechanisms",
        "description": "the specific biochemical interaction through which a drug substance produces its pharmacological effect. A mechanism of action usually includes mention of the specific molecular targets to which the drug binds, such as an enzyme or receptor.",
        "description_source": "https://en.wikipedia.org/wiki/Mechanism_of_action",
    },
    {
        "dataset": "BIOSYM",
        "name": "roas",
        "description": "the way by which a drug, fluid, poison, or other substance is taken into the body",
        "description_source": "https://en.wikipedia.org/wiki/Route_of_administration",
    },
    {
        "dataset": "BIOSYM",
        "name": "devices",
        "description": "A medical device is any device intended to be used for medical purposes.",
        "description_source": "https://en.wikipedia.org/wiki/Medical_device",
    },
    {
        "dataset": "BIOSYM",
        "name": "dosage_forms",
        "description": "Dosage forms (also called unit doses) are pharmaceutical drug products in the form in which they are marketed for use, with a specific mixture of active ingredients and inactive components (excipients), in a particular configuration (such as a capsule shell, for example), and apportioned into a particular dose.",
        "description_source": "https://en.wikipedia.org/wiki/Dosage_form",
    },
    {
        "dataset": "BIOSYM",
        "name": "behavioral_interventions",
        "description": "Non-pharmacological, non-surgical intervention such as psychotherapy, lifestyle modifications and education.",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "diagnostics",
        "description": "a medical procedure performed to detect, diagnose, or monitor diseases, disease processes, susceptibility, or to determine a course of treatment.",
        "description_source": "https://en.wikipedia.org/wiki/Medical_test",
    },
    {
        "dataset": "BIOSYM",
        "name": "research_tools",
        "description": "tools, equipment and methods for in-vitro or lab biomedical research",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "biologics",
        "description": "any pharmaceutical drug product manufactured in, extracted from, or semisynthesized from biological sources. Different from totally synthesized pharmaceuticals, they include vaccines, whole blood, blood components, allergenics, somatic cells, gene therapies, tissues, recombinant therapeutic protein, and living medicines used in cell therapy.",
        "description_source": "https://en.wikipedia.org/wiki/Biopharmaceutical",
    },
]


# compounds that inhibit ...
COMPOUND_PREFIX = (
    "(?:compound|composition)s?[ ]?(?:that (?:are|have(?: a)?)|for|of|as|which)?"
)
LOW_INFO_MOA_PREFIX = f"(?:(?:{COMPOUND_PREFIX}|symmetric|axis|binding|formula|pathway|production|receptor|(?:non )?selective|small molecule|superfamily)[ ])"
GENERIC_COMPOUND_TERM = get_or_re(COMPOUND_BASE_TERMS_GENERIC)

# e.g. "production enhancer" -> "enhancer"
# e.g. "blahblah derivative" -> "blahblah"
MOA_PATTERNS = {
    f"{LOW_INFO_MOA_PREFIX}?{pattern}(?:[ ]{GENERIC_COMPOUND_TERM})?": f" {canonical} "  # extra space removed later
    for pattern, canonical in PRIMARY_BASE_TERMS.items()
}

# inhibitory activity -> inhibitor
ACTIVITY_MOA_PATTERNS = {
    f"{pattern}(?:[ ]activity|action|function)": f" {canonical} "
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
