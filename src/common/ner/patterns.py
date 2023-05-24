"""
NER Patterns (SpaCy)

TODO:
- O-glc-NAcase
- Epiregulin/TGFα MAB
- Pirtobrutinib (LOXO-305)

"""
MOA_PATTERN = [
    # {"HEAD": { "IN": ["PROPN", "NOUN", "ADJ"] }, "OP": "+"},
    {"POS": {"IN": ["PROPN", "NOUN", "ADJ"]}, "OP": "+"},
    {
        "LOWER": {
            "IN": [
                "inhibitor",
                "dual inhibitor",
                "tri-inhibitor",
                "agonist",
                "dual agonist",
                "tri-agonist",
                "antagonist",
                "dual antagonist",
                "tri-antagonist",
                "degrader",
                "vaccine",
                "vaccination",
                "gene therapy",
                "gene transfer",
                "gene transfer therapy",
                "car-t",
                "car t",
                "chimeric antigen receptor t-cell",
                "chimeric antigen receptor t cell",
                "conjugate",
                "monoclonal antibody",
                "mab",
                "antibody",
                "antibody-drug conjugate",
                "adc",
                "tce",
                "T-cell engager",
                "bcma tce",
                "bcma t-cell engager",
                "bispecific",
                "bispecific antibody",
                "bispecific t-cell engager",
                "bte",
                "t cell engaging antibody",
                "t cell engaging antibodies",
                "prodrug",
                "pro-drug",
                "pro drug",
                "therapeutic",
                "therapeutic agent",
            ]
        },
    },
]

# pattern3 = [{"MORPH": {"IS_SUBSET": ["Number=Sing", "Gender=Neut"]}}]
