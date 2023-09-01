import unittest

from scripts.patents.utils import clean_owners


class TestPatentScriptUtils(unittest.TestCase):
    def test_clean_assignees(self):
        test_conditions = [
            {
                "terms": [
                    "Pfizer Inc",
                    "Bobs Pharmacy LLC",
                    "Bobs Pharmacy LLC inc CO",
                    "BioGen Ma",
                    "Charles River Laboratories Inc",
                    "BBOB Labs",
                    "PFIZER R&D UK LTD",
                    "Astrazeneca China",
                    "ASTRAZENECA INVEST (CHINA) CO LTD",
                    "ASTRAZENECA COLLABORATION VENTURES LLC",
                ],
                "expected_output": [
                    "Pfizer",
                    "Bobs Pharmacy",
                    "Bobs Pharmacy",
                    "Biogen",
                    "Charles River Laboratories",
                    "Bbob Laboratories",
                    "Pfizer",
                    "Astrazeneca",
                    "Astrazeneca ()",  # TODO (obvi)
                    "Astrazeneca Collaboration Ventures",  # TODO
                ],
            },
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            expected_output = condition["expected_output"]

            result = list(clean_owners(terms))
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
