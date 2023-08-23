import unittest

from scripts.patents.utils import clean_assignees


class TestPatentScriptUtils(unittest.TestCase):
    def test_clean_assignees(self):
        test_conditions = [
            {
                "terms": [
                    "Pfizer Inc",
                    "Bobs Pharmacy LLC",
                    "Bobs Pharmacy LLC INC CO",
                ],
                "expected_output": [
                    "Pfizer",
                    "Bobs Pharmacy",
                    "Bobs Pharmacy",
                ],
            },
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            expected_output = condition["expected_output"]

            result = list(clean_assignees(terms))
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
