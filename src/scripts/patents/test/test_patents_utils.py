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
                    "Astrazeneca",
                    "Astrazeneca China",
                    "ASTRAZENECA INVEST (CHINA) CO LTD",
                    "ASTRAZENECA COLLABORATION VENTURES LLC",
                    "JAPAN PHARMA CO LTD",
                    "Matsushita Electric Ind Co Ltd",
                    "US GOV CT DISEASE CONTR & PREV",
                    "US GOV NAT INST HEALTH",
                    "THE US GOV",
                    "Korea Advanced Institute Science And Technology",
                    "university of colorado at denver",
                    "university of colorado, denver",
                    "Dicerna Pharmaceuticals, Inc., a Novo Nordisk company",
                    "Genentech, Inc.",
                    "canadian institutes of health research (cihr)",
                    "agency for innovation by science and technology",
                    "janssen pharmaceuticals inc",
                ],
                "expected_output": [
                    "Pfizer",
                    "Bobs Pharmacy",
                    "Bobs Pharmacy",
                    "Biogen",
                    "Charles River Laboratories",
                    "Bbob Laboratories",
                    "Pfizer",
                    "AstraZeneca",
                    "AstraZeneca",
                    "AstraZeneca",
                    "AstraZeneca",
                    "Japan",
                    "Matsushita Electric",
                    "Government",  # TODO: US GOV!!
                    "Government",
                    "Government",
                    "Korea Advanced Institute And Technology",
                    "University Of Colorado",
                    "University Of Colorado",
                    "Dicerna",
                    "Genentech",
                    "Canadian Institutes Of Health",
                    "Agency For Innovation By And Technology",
                    "Janssen",
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
