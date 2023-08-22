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
    "activity modulator": "modulator",
    "activated protein": "protein",
    "agonist ligand": "agonist",
    "analogue": "analog",
    "antibodie": "antibody",
    "antibody construct": "antibody",
    "antibody drug": "antibody",
    "associated protein": "protein",
    "binding antagonist": "antagonist",
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
    "family member": "family",
    "family protein": "family",
    # "disease states mediated by": "associated disease", # disease states mediated by CCR5 (rearrange)
    "drug delivery": "delivery",
    "diarrhoea": "diarrhea",
    "faecal": "fecal",
    "g[ -]?protein[ -]?coupled[ -]?receptor": "g protein-coupled receptors",
    "gpcrs?": "g protein-coupled receptors",
    "homologue": "homolog",
    "inhibit(?:ing|ory?) (?:agent|compound|composition|pathway|peptide|protein|factor)": "inhibitor",
    "inhibit(?:ion|ory)?": "inhibitor",
    "kinases": "kinase",
    "mediated condition": "associated disease",
    "mediated disease": "associated disease",
    "non[ -]?steroidal": "nonsteroidal",
    "pathway inhibitor": "inhibitor",
    "protein kinase": "kinase",
    "peptide complex(?:es)?": "peptide",
    # we might want to undo this in the future (but note that the info isn't lost to the user)
    "(?:ligand )?receptor activator": "activator",
    "receptor agonist": "agonist",  # ??
    "receptor antagonist": "antagonist",  # ??
    "receptor antibody": "antibody",
    "receptor bind(?:ing|er)": "binder",
    "receptor inhibitor": "inhibitor",
    "receptor ligand": "ligand",
    "receptor modulator": "modulator",
    "receptor peptide": "peptide",
    "receptor polypeptide": "polypeptide",
    "receptor protein": "protein",
    # end
    "related condition": "associated disease",
    "related disease": "associated disease",
    "responsive protein": "protein",
    "re[ -]?uptake": "reuptake",
    "small molecule inhibitor": "inhibitor",
    "therapy agent": "therapy",
    "target(?:ing)? protein": "protein",
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
    "κ": "kappa",
    "kappa": "κ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "tau": "τ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
}
