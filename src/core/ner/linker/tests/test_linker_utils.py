import unittest

from core.ner.linker.candidate_selector.utils import (
    apply_match_retry_rewrites,
    score_candidate,
)


class TestLinkerUtils(unittest.TestCase):
    def test_apply_match_retry_rewrites(self):
        test_conditions = [
            {
                "description": "synonym test",
                "text": "Type I interferon Receptor Antagonist",
                "expected_output": "Type I interferon inhibitor",
            },
            {
                "description": "hyphenated test (short next)",
                "text": "tnf-a modulator",
                "expected_output": "tnfa modulator",
            },
            {
                "description": "hyphenated test (long next)",
                "text": "tnf-alpha modulator",
                "expected_output": "tnf alpha modulator",
            },
            {
                "description": "no rewrite",
                "text": "nothing to rewrite",
                "expected_output": None,
            },
            {
                "description": "hyphenated test (multiple)",
                "text": "glp-1 receptor-active glucagon-based peptides",
                "expected_output": "glp1 active glucagon based peptides",
            },
            {
                "description": "one word",
                "text": "glp1-agonists",
                "expected_output": "glp1 agonists",
            },
        ]

        for test in test_conditions:
            result = apply_match_retry_rewrites(test["text"])
            print("Actual", result, "expected", test["expected_output"])
            self.assertEqual(result, test["expected_output"])

    def test_score_candidate(self):
        test_conditions = [
            {
                "description": "vanilla test",
                "input": {
                    "id": "C0025202",
                    "canonical_name": "Melanoma",
                    "type_ids": ["T047"],
                    "aliases": [
                        "melanoma",
                        "melanomas",
                        "malignant melanoma",
                        "Malignant melanoma (disorder)",
                    ],
                    "matching_aliases": ["melanoma"],
                    "syntactic_similarity": 1.0,
                    "is_composite": False,
                },
                "expected_output": 1.1,
            },
            {
                "description": """
                    test of low weight match (T200 - Clinical Drug - irritatingly specific)
                    and not high match based on large alias count
                """,
                "input": {
                    "id": "C0979252",
                    "canonical_name": "paclitaxel 6 MG/ML Injectable Solution",
                    "type_ids": ["T200"],
                    "aliases": [
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
                    "matching_aliases": ["paclitaxel 6 mg/ml injectable solution"],
                    "syntactic_similarity": 1.0,
                    "is_composite": False,
                },
                "expected_output": 0.77,
            },
            {
                "description": "cui suppression test (C1704222 is suppressed)",
                "input": {
                    "id": "C1704222",
                    "canonical_name": "genome encoded entity",
                    "type_ids": ["T114"],
                    "aliases": [
                        "Genome Encoded Entity",
                        "gene product",
                        "Genome Encoded Entity",
                    ],
                    "matching_aliases": [],
                    "syntactic_similarity": 1.0,
                    "is_composite": False,
                },
                "expected_output": 0.0,
            },
            {
                "description": "name suppression test ('rat' is suppressed)",
                "input": {
                    "id": "C1566744",
                    "canonical_name": "CYP2F4, rat",
                    "type_ids": ["T116"],
                    "aliases": [
                        "CYP2F4, rat",
                        "cytochrome P-450 2F4, rat",
                        "Cyp2f2 protein, rat",
                    ],
                    "matching_aliases": [],
                    "syntactic_similarity": 1.0,
                    "is_composite": False,
                },
                "expected_output": 0.0,
            },
            {
                "description": "non-composite/inhibitor (conditionally suppressed cui)",
                "input": {
                    "id": "C1999216",
                    "canonical_name": "inhibitors",
                    "type_ids": ["T123"],
                    "aliases": [
                        "Inhibitors",
                        "inhibitor",
                    ],
                    "matching_aliases": ["inhibitor"],
                    "syntactic_similarity": 0.8,
                    "is_composite": False,
                },
                "expected_output": 0.0,
            },
            {
                "description": "composite/inhibitor (conditionally suppressed cui)",
                "input": {
                    "id": "C1999216",
                    "canonical_name": "inhibitors",
                    "type_ids": ["T123"],
                    "aliases": [
                        "Inhibitors",
                        "inhibitor",
                    ],
                    "matching_aliases": ["inhibitor"],
                    "syntactic_similarity": 0.8,
                    "is_composite": True,
                },
                "expected_output": 0.88,
            },
        ]

        for condition in test_conditions:
            expected_output = condition["expected_output"]

            result = score_candidate(**condition["input"])
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
