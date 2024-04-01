import unittest

from nlp.nel.candidate_selector.utils import apply_match_retry_rewrites


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
