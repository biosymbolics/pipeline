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

PHRASE_MAP = {
    "activity inhibitor": "inhibitor",
    "activity modulator": "modulator",
    "activity antagonist": "antagonist",
    "activity agonist": "agonist",
    "activation agonist": "agonist",
    "activation antagonist": "antagonist",
    "activation inhibitor": "inhibitor",
    "activation modulator": "modulator",
    "activated protein": "protein",
    "agonist ligand": "agonist",
    "analogue": "analog",
    "activating": "activator",
    "activator activity": "activator",
    "activator action": "activator",
    "antibody conjugate": "antibody",
    "antibody immunoconjugate": "antibody",
    "antibodie": "antibody",
    "antibody construct": "antibody",
    "antibody drug": "antibody",
    "associated protein": "protein",
    "associated illness": "associated disease",
    "axis antagonist": "antagonist",
    "axis inhibitor": "inhibitor",
    "axis modulator": "modulator",
    "axis receptor": "receptor",
    "axis agonist": "agonist",
    "axis protein": "protein",
    "axis peptide": "peptide",
    "binding antagonist": "antagonist",
    "binding antibody": "antibody",
    "binding protein": "protein",
    "binding modulator": "modulator",
    "binding activity": "binder",
    "binding agent": "binder",
    "binding region": "binder",
    "biologic(?:al)? response modifiers?": "immunomodulator",
    "blocking agent": "blocker",
    "blockade": "blocker",
    # chimeric antibody-T cell receptor
    "chimeric[ -]?(?:antigen|antibody)[ -]?receptor": "chimeric antigen receptor",
    "chimeric[ -]?(?:antigen|antibody)[ -]?(?:t[ -]?cell ) receptor": "chimeric antigen receptor t-cell",
    "car[ -]t": "chimeric antigen receptor t-cell",
    "compound inhibiting": "inhibitor",
    "conditions and disease": "diseases",
    "disease factors": "diseases",
    "diseases and disorder": "diseases",
    "disorders and disease": "diseases",
    "disease state": "diseases",
    "diseases and condition": "diseases",
    "induced diseases": "diseases",
    "family member": "family",
    "family protein": "family",
    # "disease states mediated by": "associated disease", # disease states mediated by CCR5 (rearrange)
    "drug delivery": "delivery",
    "diarrhoea": "diarrhea",
    "faecal": "fecal",
    "g[ -]?protein[ -]?coupled[ -]?receptor": "g protein-coupled receptors",
    "gpcrs?": "g protein-coupled receptors",
    "homologue": "homolog",
    "ifn": "interferon",
    "inhibit(?:ing|ory?) (?:agent|compound|composition|pathway|peptide|protein|factor)": "inhibitor",
    "inhibit(?:ion|ory)?": "inhibitor",
    "kinases": "kinase",
    "mediated condition": "associated disease",
    "mediated disease": "associated disease",
    "non[ -]?steroidal": "nonsteroidal",
    "pathway inhibitor": "inhibitor",
    "pathway modulator": "modulator",
    "pathway protein": "protein",
    "pathway receptor": "receptor",
    "pathway agonist": "agonist",
    "pathway antagonist": "antagonist",
    "pathway peptide": "peptide",
    "pathway polypeptide": "polypeptide",
    "pathway degrad[a-z]{1,5}": "degrader",
    "protein kinase": "kinase",
    "protein degrader": "degrader",
    "peptide complex(?:es)?": "peptide",
    "(?:poly)?peptide chain": "polypeptide",
    # we might want to undo this in the future (but note that the info isn't lost to the user)
    "(?:ligand )?receptor activator": "activator",
    "receptor agonist": "agonist",  # ??
    "receptor antagonist": "antagonist",  # ??
    "receptor antibody": "antibody",
    "receptor activat(?:ion|or)": "activator",
    "receptor bind(?:ing|er)": "binder",
    "receptor inhibitor": "inhibitor",
    "receptor ligand": "ligand",
    "receptor modulator": "modulator",
    "receptor peptide": "peptide",
    "receptor polypeptide": "polypeptide",
    "receptor protein": "protein",
    "receptor degrad[a-z]{1,5}": "degrader",
    # end
    "related condition": "associated disease",
    "related disease": "associated disease",
    "responsive protein": "protein",
    "re[ -]?uptake": "reuptake",
    "(?:non )?selective inhibitor": "inhibitor",
    "(?:non )?selective modulator": "modulator",
    "(?:non )?selective antagonist": "antagonist",
    "(?:non )?selective agonist": "agonist",
    "(?:non )?selective peptide": "peptide",
    "(?:non )?selective protein": "protein",
    "(?:non )?selective receptor": "receptor",
    "small molecule inhibitor": "inhibitor",
    "therapy agent": "therapy",
    "therapeutic agent": "therapy",
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
    # IGF-I receptor subunits
    "PEG": "pegylated",
}
