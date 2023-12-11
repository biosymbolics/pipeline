"""
Constants related to Biologic drugs
"""

CELL_THERAPY_PRODUCT_INFIXES = {
    "den": "dendritic cells",
    "mio": "myoblasts",
    "co": "chondrocytes",
    "fi": "fibroblasts",
    "ker": "keratinocytes",
    "end": "endothelial cells",
    "leu": "lymphocytes/monocytes/APC (white cells)",
    "cima": "cytosine deaminase",
    "ermin": "growth factor",
    "kin": "interleukin",
    "lim": "immunomodulator",
    "lip": "human lipoprotein lipase",
    "mul": "multiple gene",
    "stim": "colony stimulating factor",
    "tima": "thymidine kinase",
    "tusu": "tumour suppression",
}

MONOCLONAL_ANTIBODY_INFIXES = {
    "xizu": "chimeric-humanized",
    "xi": "chimeric",
    "zu": "humanized",
    "vet": "veterinary use",
    "a": "rat",
    "axo": "rat-mouse",
    "e": "hamster",
    "i": "primate",
    "o": "mouse",
    "u": "human",
}

TARGET_INFIXES = {
    "ba": "bacterial",  # antibody targeting bacteria
    "ami": "serum amyloid protein (SAP)/amyloidosis",  # antibody targeting serum amyloid protein (SAP)/amyloidosis
    "ci": "cardiovascular",  # antibody targeting cardiovascular system
    "fu": "fungal",  # antibody targeting fungi
    "gro": "skeletal muscle mass related growth factors and receptors",  # antibody targeting skeletal muscle mass related growth factors and receptors
    "ki": "interleukin",  # antibody targeting interleukins
    "li": "immunomodulating",  # antibody targeting immunomodulating agents
    "ne": "neural",  # antibody targeting neural system
    "so": "bone",  # antibody targeting bone
    "tox": "toxin",  # antibody targeting toxins
    "tu": "tumour",  # antibody targeting tumours
    "vi": "viral",  # antibody targeting viruses
}

# https://www.uspharmacist.com/article/naming-of-biological-products
BIOLOGIC_SUFFIXES = {
    "pressin": "vasocontrictors and vasopressin derivatives",
    "rsen": "antisense nucleotides",
    "cel": "cell therapies",
    "cept": "indicates a protein that mimics an immunoglobulin",  # luspatercept-aamt??
    "stim": "colony stimulating factors",
    # "ase": "enzymes",  # FP: disease
    "som": "growht hormone derivatives",
    "cogin": "Blood coagulation cascade inhibitors",
    "cog": "Blood coagulation factors",
    "gene": "e.g. lisocabtagene maraleucel",
    "relix": "Gonadotropin-releasing hormone (GnRH) inhibiting peptides",
    "ermin": "Growth factors and Tumour necrosis factors (TNF)",  # FP:  # determined, determination
    "parin": "Heparin derivatives",
    "irudin": "Hirudin derivatives",
    "imod": "Immunomodulators",
    "kinra": "Interleukin receptor antagonists ",
    # "kin": "Interleukin type substances",  # FP: Hodgkin
    "mab": "mAb",
    "tocin": "Oxytocin derivatives",
    "relin": "Pituitary hormone-release stimulating peptides",
    "cept": "Receptor molecules, native or modified",
    "mide": "e.g. lenalidomide, mezigdomide",
    "mod": "e.g. ozanimod",
    "tide": "",
    "actide": "Synthetic polypeptides with a corticotropin-like action",
}

BIOLOGIC_INFIXES = {
    # "apt": "aptamers", # FP: chapters
    "siran": "Small interfering RNA ",
}


BIOLOGIC_BASE_TERMS: list[str] = [
    "(?:(?:glyco|fusion)[ -]?)?proteins?(?: binding)?",
    "(?:poly)peptides?(?: binding)?",
    "(?:poly|di|tri)?nucleo[ts]ides?(?: binding)?",
    "(?:receptor |protein )?ligands?(?: binding| that bind)?",
    "(?:dna |nucleic acid |antibody )?(?:fragments?|sequences?)(?: binding| that bind)?",
    "aggregate",
    "allele",
    "(?:amino |fatty |(?:poly|di|tri)?nucleic )acid",
    "(?:neo)?antigen",
    "antigen[ -]binding fragment",
    "antigen presenting cell",
    "aptamer",
    "anion",
    "(?:auto)?antibod(?:ie|y)",
    "adc",
    "antibod(?:y|ie)",
    "antibody[ -]?drug[ -]conjugate",
    "(?:antibody |dna |rna )?molecules?(?: binding)?",
    "bio(?:material|chemical)",
    "biosimilar",
    "bcma(?: nke)?",
    "car[- ]?(?:t|nk)",
    "chimeric antigen receptor",
    "chemotherap(?:y|ie)",  # not necessary biologic, more class/descriptive
    "(?:adoptive )?cell(?: tranfer)?(?: therap(?:y|ie))?",  # cell disease
    "clone",
    "complement(?: component)?",
    "conjugate",
    "(?:cell )?culture",
    "cytokine",
    "decomposition",
    "(?:recombinant )?dna",
    "epitope",
    "exosome",
    "enzyme",
    "(?:gene )?expression",
    "fab(?: region)?",
    "factor [ivx]{1,3}",
    "fc(?:[- ]fusion )?(?: protein)?",
    "fus(?:ed|ion)",
    "gene",
    "(?:growth )?factor",
    "(?:hairpin|micro|messenger|sh|si|i|ds|m|hp|recombinant|double[- ]?stranded|small[- ]?interfering|guide)?[ -]?(?:rna|ribonucleic acid)(?: molecule| sequence)?",
    "(?:trans)?gene",
    "hormone",
    "immun[io][zs](?:ation|ing|e|logical|therapy|genic)",
    "insulin",
    "interferon",
    "interleukin",
    "isoform",
    "(?:phospho[ -]?|sphingo[ -]?)?lipid",
    "keratin",
    "(?:dehydratase|esterase|hydrolase|ligase|mutase|oxidase|protease|synthase|transferase|transcriptase|reductase|kinase|nuclease)",  # some enzymes
    "(?:co[ -]?)?(?:di|mono|poly|oligo)mer",
    "(?:natural killer|nkt|nkc)(?: cells?)?",
    "lignin",
    "liposome",
    "(?:macro|bio)?[ -]?molecule",
    "(?:(?:neuro|immuno)[- ]?)?modulat(?:or|ion|ing|e)s?(?: binding)?",
    "motif",
    "mutein",
    "mutation",
    "(?:oligo[ -])?nucleotides?(?: sequence)?",
    "(?:poly[ -]?)?peptides?(?: sequence)?",
    "plasmid",
    "protein",
    "recombinant",
    "receptor",  # ??
    "recognizer",
    "(?:recombinant|attenuated|antibod(?:y|ies)|inhibit|anti|chimeric|synthetic).*virus",
    "scaffold",
    "scfv",  # single-chain variable fragment
    "sequence",
    "(?:expression )?vector",
    "stem cell",
    "strand",
    "toxin",
    "(?:tumor[- ]?infiltrating )?lymphocyte",
    "(?:(?:immuno|gene)[ -]?)?therap(?:y|ie)",
    "transcription(?: factor)?",
    "(?:(?:[t|b][ -]cell )engag(?:er|ing|e|ment)(?: receptor)?|tce(?:r))",
]
