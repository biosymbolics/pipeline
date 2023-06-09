import unittest

import spacy
from common.ner.utils import remove_common_terms


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
        vocab = spacy.load("en_core_web_sm").vocab
        test_conditions = [
            {
                "terms": [
                    "vaccine candidates",
                    "PF-06863135",
                    "therapeutic options",
                    "COVID-19 treatments",
                    "exception term",
                    "common term",
                ],
                "exception_list": ["exception"],
                "expected_output": [
                    "PF-06863135",
                    "COVID-19 treatments",
                    "exception term",
                ],
            },
            {
                "terms": [
                    "vaccine candidates",
                    "therapeutic options",
                    "COVID-19 treatments",
                    "common term",
                ],
                "exception_list": [],
                "expected_output": ["COVID-19 treatments"],
            },
            # Add more test conditions as needed
        ]

        for condition in test_conditions:
            terms = condition["terms"]
            exception_list = condition["exception_list"]
            expected_output = condition["expected_output"]

            result = remove_common_terms(vocab, terms, exception_list)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
