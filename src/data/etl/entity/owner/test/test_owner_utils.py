import unittest

from prisma.enums import OwnerType

from data.etl.entity.owner.owner import generate_clean_owner_map, OwnerTypeParser


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
                "name": "random l.l.c.",
                "expected_output": OwnerType.INDUSTRY,
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
                    "pfizer",
                    "bobs pharmacy",
                    "bobs pharmacy",
                    "biogen",
                    "charles river laboratories",
                    "bbob laboratories",
                    "pfizer",
                    "astrazeneca",
                    "astrazeneca",
                    "astrazeneca",
                    "astrazeneca",
                    "japan",  # TODO
                    "matsushita electric",
                    "us government",
                    "us government",
                    "us government",
                    "korea advanced institute and technology",
                    "university of colorado",
                    "university of colorado",
                    "dicerna",
                    "genentech",
                    "canadian institutes of health",
                    "agency for innovation by and technology",
                    "janssen",
                ],
            },
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            expected_output = condition["expected_output"]

            owner_map = generate_clean_owner_map(terms)
            result = [owner_map.get(term) or term for term in terms]
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
