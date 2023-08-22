MECHANISM_BASE_TERMS: list[str] = [
    "activity",
    "adjuvant",
    "agent",
    "amplifier",
    "analgesic",
    "anaesthetic",
    "anti[ -]?microbial",
    "anti[ -]?biotic" "anti[ -]?infective",
    "anti[ -]?inflammatory",
    "anti[ -]?bacterial",
    "anti[ -]?viral",
    "anti[ -]?metabolite",
    "carrier",
    "catalyst",
    "contraception",
    "co[- ]?factor",
    "(?:cross[ -]?)?linker",
    "diuretic",
    "donor",
    "differentiation",
    "dilator",
    "disinfectant",
    "effect",
    "emitter",
    "emollient",
    "emulsifier",
    "encoder",
    "expression vectors",
    "factor",
    "growth factor",
    "interference",
    "introducer",
    "interaction",
    "lubricant",
    "metabolite",
    "mimetic",
    "(?:oligo[ -])?nucleotide",
    "phototherapy",
    "precursor",
    "primer",
    "pro[ -]?drug",
    "pathway",
    "solvent",
    "surfactant",
    "transfer",
    "toxin",
    "vaccine",
    "vaccine adjuvant",
]
BIOLOGIC_BASE_TERMS: list[str] = [
    "antigen",
    "antigen[ -]binding fragment",
    "aptamer",
    "anion",
    "antibodie",
    "antibody",
    "autoantibody",
    "autoantibodie",
    "ADC",
    "antibody[ -]?drug conjugate",
    "chemotherap(?:y|ie)",  # not necessary biologic, more class/descriptive
    "cell",
    "clone",
    "decomposition",
    "dimer",
    "exosome",
    "enzyme",
    "hairpin rna",
    "ion",
    "isoform",
    "(?:phospho[ -]?)?lipid",
    "kinase",
    "micro[ ]?rna",
    "messenger rna",
    "monomer",
    "mrna",
    "neoantigen",
    "nuclease",
    "oligomer",
    "polymer",
    "protease",
    "rna",
    "sh[ ]?rna",
    "si[ ]?rna",
    "ds[ ]?rna",
    "double[- ]?stranded rna",
    "sequence",
    "scfv",  # single-chain variable fragment
    "liposome",
    "vector",
    "stem cell",
    "gene therap(?:y|ie)",
    "immunotherap(?:y|ie)",
]
COMPOUND_BASE_TERMS: list[str] = [
    "acid",
    "amide",
    "amine",
    "analog",
    "analogue",
    "aerosol",
    "capsule",
    "carbonyl",
    "composition",
    "compound",
    "cream",
    "derivative",
    "drug",
    "drop",
    "element",
    "ester",
    "form",
    "formulation",
    "gel",
    "indole",
    "inhalant",
    "injection",
    "group",
    "homologue",
    "hydrogel",
    "microparticle",
    "nanoparticle",
    "nutraceutical",
    "ointment",
    "ortholog",
    "particle",
    "patch",
    "pill",
    "polyamide",
    "polymer",
    "powder",
    "reagent",
    "regimen",
    "solution",
    "spray",
    "stereoisomer",
    "suppository",
    "tablet",
    "therapy",
    "therapie",
]

INTERVENTION_BASE_TERMS = [
    *BIOLOGIC_BASE_TERMS,
    *MECHANISM_BASE_TERMS,
    *COMPOUND_BASE_TERMS,
]

MECHANISM_BASE_TERM_SETS: list[list[str]] = [
    [
        "immunization",
        "immunizing",
        "immunize",
        "immunization",
        "immunological",
        "immunoconjugate",
        "immunotherapy",
        "immunogenic compositions",
    ],
    ["regenerate", "regeneration", "regenerating", "regeneration"],
    ["stabilizer", "stabilizing", "stabilize", "stabilization"],
    ["modifier", "modifying", "modification"],
    ["inhibitor", "inhibition", "inhibiting", "inhibit"],
    ["agonist", "agonizing", "agonize", "agonism"],
    ["antagonist", "antagonizing", "antagonize", "antagonism"],
    [
        "activator",
        "activation",
        "activating",
    ],
    [
        "(?:de)?sensitizer",
        "(?:de)?sensitization",
        "(?:de)?sensitizing",
        "(?:de)?sensitize",
    ],
    ["suppressor", "suppression", "suppressing", "suppress"],
    ["stimulator", "stimulation", "stimulating", "stimulate"],
    [
        "(?:(?:neuro|immuno)[- ]?)?modulate",
        "(?:(?:neuro|immuno)[- ]?)?modulates? binding",
        "(?:(?:neuro|immuno)[- ]?)?modulating",
        "(?:(?:neuro|immuno)[- ]?)?modulation",
        "(?:(?:neuro|immuno)[- ]?)?modulator",
    ],
    ["promotion", "promoting", "promote"],
    ["enhancement", "enhancing", "enhance", "enhancer"],
    ["regulator", "regulation", "regulating"],
    ["degrader", "degradation", "degrading", "degrade"],
    ["inducer", "inducing", "induce", "induction"],
    ["promoter", "promoting", "promote", "promotion"],
    ["blocker", "blockade", "blocking", "block"],
    ["transporter", "transporting", "transport", "transportation"],
]

BIOLOGIC_BASE_TERM_SETS: list[list[str]] = [
    ["(?:glyco[ -]?)?protein", "(?:glyco[ -]?)?proteins? binding"],
    ["(?:poly)peptide", "(?:poly)peptides? binding"],
    ["(?:poly)?nucleotide", "(?:poly)?nucleotides? binding"],
    ["ligand", "ligands? binding", "ligands? that bind"],
    [
        "(?:dna |nucleic acid)?fragment",
        "(?:dna |nucleic acid)?fragments? binding",
        "(?:dna |nucleic acid)?fragments? that bind",
    ],
]
COMPOUND_BASE_TERM_SETS: list[list[str]] = [
    ["conjugation", "conjugating", "conjugate"],
    ["(?:small |antibody )?molecule", "(?:small |antibody )?molecules? binding"],
]

INTERVENTION_BASE_TERM_SETS: list[list[str]] = [
    *MECHANISM_BASE_TERM_SETS,
    *BIOLOGIC_BASE_TERM_SETS,
    *COMPOUND_BASE_TERM_SETS,
]

INTERVENTION_BASE_PREFIXES = [
    "acceptable",
    "allosteric",
    "anti",
    "aromatic",
    "bi-?specific",
    "chimeric",
    "cyclic",
    "dual",
    "encoding",
    "(?:bi[- ])?functional",
    "heterocyclic",
    "inverse",
    "irreversible",
    "(?:ion )?channel",
    "inventive",
    "monoclonal",
    "negative",
    "novel",
    "new",
    "partial",
    "polyclonal",
    "positive",
    "potent",
    "optionally",
    "receptor",
    "reversible",
    "recombinant",
    "topical",
    "tricyclic",
    "selective",
    "single",
    "short",
    "substituted",
]
