import unittest

from data.common.biomedical.umls import clean_umls_name


UMLS_NAME_OVERRIDES = {
    "C4721408": "Antagonist",  # "Substance with receptor antagonist mechanism of action (substance)"
    "C0005525": "Modulator",  # Biological Response Modifiers https://uts.nlm.nih.gov/uts/umls/concept/C0005525
    "C1145667": "Binder",  # https://uts.nlm.nih.gov/uts/umls/concept/C1145667
}


class TestUmlsUtils(unittest.TestCase):
    def setUp(self):
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
                "description": "choose canonical name if 2 words",
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
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = clean_umls_name(
                test["cui"], test["canonical_name"], test["aliases"]
            )
            self.assertEqual(result, expected_output)
