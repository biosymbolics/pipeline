import unittest

from prisma.enums import OwnerType

from data.etl.entity.owner.owner import clean_owners, OwnerTypeParser


class TestPatentScriptUtils(unittest.TestCase):
    def test_owner_type_parser(self):
        test_conditions = [
            {
                "name": "mayo clinic",
                "expected_output": OwnerType.HEALTH_SYSTEM,
            },
            {
                "name": "albert einstein college medicine",
                "expected_output": OwnerType.UNIVERSITY,
            },
            {
                "name": "wisconsin university",
                "expected_output": OwnerType.UNIVERSITY,
            },
            {
                "name": "pfizer",
                "expected_output": OwnerType.INDUSTRY_LARGE,
            },
            {
                "name": "random co, inc",
                "expected_output": OwnerType.INDUSTRY,
            },
            {
                "name": "us government",
                "expected_output": OwnerType.GOVERNMENTAL,
            },
            {
                "name": "bristol university research foundation",
                "expected_output": OwnerType.UNIVERSITY,
            },
        ]

        for test in test_conditions:
            name = test["name"]
            expected_output = test["expected_output"]

            result = OwnerTypeParser.find(name)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

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
