import unittest

from common.ner.cleaning import filter_common_terms, normalize_entity_names


class TestNerUtils(unittest.TestCase):
    def test_filter_common_terms(self):
        test_conditions = [
            {
                "terms": [
                    "vaccine candidates",
                    "PF-06863135",
                    "therapeutic options",
                    "COVID-19 mRNA vaccine",
                    "exception term",
                    "common term",
                ],
                "exception_list": ["exception"],
                "expected_output": [
                    "PF-06863135",
                    "COVID-19 mRNA vaccine",
                    "exception term",
                ],
            },
            {
                "terms": [
                    "vaccine candidate",
                    "vaccine candidates",
                    "therapeutic options",
                    "therapeutic option",
                    "therapeutics option",
                    "COVID-19 mRNA vaccine",
                    "common term",
                ],
                "exception_list": [],
                "expected_output": ["COVID-19 mRNA vaccine"],
            },
            # Add more test conditions as needed
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            exception_list = condition["exception_list"]
            expected_output = condition["expected_output"]

            result = filter_common_terms(terms, exception_list)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

    def test_clean_entities(self):
        test_conditions = [
            {
                "input": "OPSUMIT ®",
                "expected": "opsumit",
            },
            {
                "input": "OPSUMIT®",
                "expected": "opsumit",
            },
            {
                "input": "OPSUMIT®, other product",
                "expected": "opsumit, other product",
            },
            {
                "input": "/OPSUMIT ®",
                "expected": "OPSUMIT",
            },
            {
                "input": "5-ht1a inhibitors",
                "expected": "5-ht1a inhibitor",
            },
            {
                "input": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivatives",
                "expected": "1-(3-aminophenyl)-6,8-dimethyl-5-(4-iodo-2-fluoro-phenylamino)-3-cyclopropyl-1h,6h-pyrido[4,3-d]pyridine-2,4,7-trione derivative",
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = normalize_entity_names([input])
            print("Actual", result, "expected", [expected])
            self.assertEqual(result, [expected])


if __name__ == "__main__":
    unittest.main()
