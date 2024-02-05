import unittest

from core.ner.linker.utils import score_candidate


class TestLinkerUtils(unittest.TestCase):
    def test_score_candidate(self):
        test_conditions = [
            {
                "description": "vanilla test",
                "input": {
                    "candidate_id": "C0025202",
                    "candidate_canonical_name": "Melanoma",
                    "candidate_type_ids": ["T047"],
                    "candidate_aliases": [
                        "melanoma",
                        "melanomas",
                        "malignant melanoma",
                        "Malignant melanoma (disorder)",
                    ],
                    "syntactic_similarity": 1.0,
                },
                "expected_output": 1.1,
            },
            {
                "description": """
                    test of low weight match (T200 - Clinical Drug - irritatingly specific)
                    and not high match based on large alias count
                """,
                "input": {
                    "candidate_id": "C0979252",
                    "candidate_canonical_name": "paclitaxel 6 MG/ML Injectable Solution",
                    "candidate_type_ids": ["T200"],
                    "candidate_aliases": [
                        "paclitaxel 100 mg per 16.7 ml injectable solution",
                        "paclitaxel 100mg/17ml inj,conc",
                        "paclitaxel 100mg/17ml vil inj,conc",
                        "paclitaxel 150 mg per 25 ml injectable solution",
                        "paclitaxel 30 mg per 5 ml injectable solution",
                        "paclitaxel 300 mg in 50 ml intravenous injection, solution [paclitaxel]",
                        "paclitaxel 300 mg per 50 ml injectable solution",
                        "paclitaxel 300mg/50ml inj vil 50ml",
                        "paclitaxel 300mg/50ml inj,vil,50ml",
                        "paclitaxel 30mg/5ml inj,conc",
                        "paclitaxel 30mg/5ml vil inj,conc",
                        "paclitaxel 6 mg in 1 ml intravenous injection",
                        "paclitaxel 6 mg in 1 ml intravenous injection, solution",
                        "paclitaxel 6 mg in 1 ml intravenous injection, solution [paclitaxel paclitaxel]",
                        "paclitaxel 6 mg in 1 ml intravenous injection, solution [paclitaxel]",
                        "paclitaxel 6 mg in 1 ml intravenous injection, solution, concentrate",
                        "paclitaxel 6 mg/ml injectable solution",
                        "paclitaxel 6mg/ml inj,conc,iv",
                    ],
                    "syntactic_similarity": 1.0,
                },
                "expected_output": 0.77,
            },
            {
                "description": "cui suppression test (C1704222 is suppressed)",
                "input": {
                    "candidate_id": "C1704222",
                    "candidate_canonical_name": "genome encoded entity",
                    "candidate_type_ids": ["T114"],
                    "candidate_aliases": [
                        "Genome Encoded Entity",
                        "gene product",
                        "Genome Encoded Entity",
                    ],
                    "syntactic_similarity": 1.0,
                },
                "expected_output": 0.0,
            },
            {
                "description": "name suppression test ('rat' is suppressed)",
                "input": {
                    "candidate_id": "C1566744",
                    "candidate_canonical_name": "CYP2F4, rat",
                    "candidate_type_ids": ["T116"],
                    "candidate_aliases": [
                        "CYP2F4, rat",
                        "cytochrome P-450 2F4, rat",
                        "Cyp2f2 protein, rat",
                    ],
                    "syntactic_similarity": 1.0,
                },
                "expected_output": 0.0,
            },
        ]

        for condition in test_conditions:
            expected_output = condition["expected_output"]

            result = score_candidate(**condition["input"])
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
