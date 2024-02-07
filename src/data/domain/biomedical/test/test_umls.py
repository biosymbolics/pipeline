import unittest

from data.domain.biomedical.umls import clean_umls_name, is_umls_suppressed


UMLS_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
    "C0005525": "Modulator",  # Biological Response Modifiers https://uts.nlm.nih.gov/uts/umls/concept/C0005525
    "C1145667": "Binder",  # https://uts.nlm.nih.gov/uts/umls/concept/C1145667
}


class TestUmlsUtils(unittest.TestCase):
    def setUp(self):
        pass

    def test_is_umls_suppressed(self):
        test_cases = [
            {
                "description": "is suppressed due CUI",
                "cui": "C1704222",
                "canonical_name": "gene encoded entity",
                "matching_aliases": [],
                "expected": True,
            },
            {
                "description": "is suppressed based on name",
                "cui": "C5197009",
                "canonical_name": "ZEB1 protein, rat",
                "matching_aliases": ["ZEB-1 protein, rat"],
                "expected": True,
            },
            {
                "description": "is suppressed due to synonym",
                "cui": "C0025611",
                "canonical_name": "methamphetamine",
                "matching_aliases": ["ice"],
                "expected": True,
            },
            {
                "description": "not suppressed since first synonym is not suppressed",
                "cui": "C0025611",
                "canonical_name": "methamphetamine",
                "matching_aliases": ["methamphetamine", "amphetamine", "ice"],
                "expected": False,
            },
        ]

        for test in test_cases:
            result = is_umls_suppressed(
                test["cui"], test["canonical_name"], test["matching_aliases"]
            )
            self.assertEqual(result, test["expected"])

    def test_tuis_to_entity_type(self):
        # see test_CanonicalEntity_type
        pass

    def test_clean_umls_name(self):
        test_cases = [
            {
                "description": "prefer shorter names, with same first letter",
                "cui": "abcdnx",
                "canonical_name": "GPR84 gene",
                "aliases": [
                    "GPR84 gene",
                    "EX33",
                    "GPR84",
                ],
                "expected": "GPR84",
            },
            {
                "description": "prefer no comma",
                "cui": "C0968164",
                "canonical_name": "GPR84 protein, human",
                "aliases": [
                    "GPR84 protein, human",
                    "EX33 protein, human",
                    "GPR84 a human protein",  # fabricated alias
                    "G protein-coupled receptor 84, human",
                ],
                "expected": "GPR84 a human protein",
            },
            {
                "description": "prefer no [Meta]",
                "cui": "C2757011",
                "canonical_name": "Tyrosine Kinase Inhibitors [MoA]",
                "aliases": [
                    "Tyrosine Kinase Inhibitors [MoA]",
                    "Tyrosine Kinase Inhibitors",
                    "Tyrosine Kinase Inhibitors [MoA]",
                ],
                "expected": "Tyrosine Kinase Inhibitors",
            },
            {
                "description": "do not choose ridiculously short alias",
                "cui": "C0003873",
                "canonical_name": "Rheumatoid Arthritis (RA)",  # fabricated to avoid other rule
                "aliases": [
                    "Rheumatoid Arthritis",
                    "RA",
                    "Rheumatoid arthritis (disorder)",
                ],
                "expected": "Rheumatoid Arthritis",
            },
            {
                "description": "avoid alias with ()",
                "cui": "C5200928",
                "canonical_name": "High (finding)",
                "aliases": [
                    "High (finding)",
                    "High",
                ],
                "is_composite": True,
                "expected": "High",
            },
            {
                "description": "choose canonical name if short-ish (5 words or fewer)",
                "cui": "C0003873",
                "canonical_name": "Rheumatoid Arthritis",
                "aliases": [
                    "Rheum. Arthritis",
                    "RA",
                    "Rheumatoid arthritis (disorder)",
                ],
                "expected": "Rheumatoid Arthritis",
            },
            {
                "description": "don't choose canonical name if 2 words but second is (gene or protein)",
                "cui": "C1415265",
                "canonical_name": "GPR84 gene",
                "aliases": [
                    "GPR84 gene",
                    "EX33",
                    "GPR84",
                ],
                "expected": "GPR84",
            },
            {
                "description": "apply overrides",
                "cui": "C4721408",
                "canonical_name": "blah blah xyz abc",
                "aliases": [
                    "grp12",
                    "xx122",
                    "happy fake example",
                ],
                "expected": "Antagonist",
            },
            {
                "description": "is not composite, still apply overrides",
                "cui": "C4721408",
                "canonical_name": "blah blah xyz abc",
                "aliases": [
                    "grp12",
                    "xx122",
                    "happy fake example",
                ],
                "is_composite": False,
                "expected": "Antagonist",
            },
            {
                "description": "if isn't composite, prefer canonical name",
                "cui": "C0968164",
                "canonical_name": "GPR84 protein, human",
                "aliases": [
                    "GPR84 protein, human",
                    "EX33 protein, human",
                    "GPR84 a human protein",  # fabricated alias
                    "G protein-coupled receptor 84, human",
                ],
                "is_composite": False,
                "expected": "GPR84 protein, human",
            },
            {
                "description": "...unless it's a gene/protein",
                "cui": "C0061355",
                "canonical_name": "Glucagon-Like Peptide 1",
                "aliases": [
                    "Glucagon-Like Peptide 1",
                    "GLP-1",
                    "Glucagon-like peptide 1 (substance)",
                ],
                "type_ids": ["T116"],
                "is_composite": False,
                "expected": "GLP-1",
            },
            {
                "description": "...unless it is a stupidly long name",
                "cui": "C4086713",
                "canonical_name": "Substance with programmed cell death protein 1 inhibitor mechanism of action (substance)",
                "aliases": [
                    "Programmed Cell Death Protein 1 Inhibitors",
                    "PD1 Inhibitor",
                    "PD-1 Inhibitor",
                ],
                "is_composite": False,
                "expected": "PD1 Inhibitor",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = clean_umls_name(
                test["cui"],
                test["canonical_name"],
                test["aliases"],
                test.get("type_ids", []),
                test.get("is_composite", True),
            )
            self.assertEqual(result, expected_output)
