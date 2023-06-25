import unittest

import spacy
from clients.spacy import Spacy

from common.ner.cleaning import remove_common_terms, clean_entity


class TestNerUtils(unittest.TestCase):
    # def test_remove_unmatched_brackets(self):
    #     test_patterns = [
    #         {"text": "Omicron BA.4/BA.5)/ Comirnaty Original/Omicron BA.1 Vaccine", "expected": "Omicron BA.4/BA.5/ Comirnaty Original/Omicron BA.1 Vaccine"},
    #         {"text": "Example (with unmatched)", "expected": "Example (with unmatched)"},
    #         {"text": "Another {example] with unmatched", "expected": "Another example with unmatched"},
    #     ]

    #     for pattern in test_patterns:
    #         text = pattern["text"]
    #         expected_output = pattern["expected"]
    #         result = remove_unmatched_brackets(text)
    #         self.assertEqual(result, expected_output)

    def test_remove_common_terms(self):
        nlp = Spacy.get_instance("en_core_web_sm", disable=["ner"])
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

            result = remove_common_terms(terms, nlp, exception_list)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)

    def test_clean_entities(self):
        test_conditions = [
            {
                "input": "OPSUMIT ®",
                "expected": "OPSUMIT",
            },
            {
                "input": "OPSUMIT®",
                "expected": "OPSUMIT",
            },
            {
                "input": "OPSUMIT®, other product",
                "expected": "OPSUMIT, other product",
            },
            {
                "input": "/OPSUMIT ®",
                "expected": "OPSUMIT",
            },
        ]

        for condition in test_conditions:
            input = condition["input"]
            expected = condition["expected"]

            result = clean_entity(input)
            print("Actual", result, "expected", expected)
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
