import unittest

from data.ner import NerTagger


class TestNerUtils(unittest.TestCase):
    """
    from data.ner import NerTagger; tagger=NerTagger()
    t = tagger.extract([text], link=False)[0]
    [(t1[0], t1.start_char, t1.end_char) for t1 in t]
    """

    def setUp(self):
        self.tagger = NerTagger(
            entity_types=frozenset(["compounds", "diseases", "mechanisms"])
        )

    def test_ner(self):
        test_conditions = [
            {
                "text": """
                Bioenhanced formulations comprising eprosartan in oral solid dosage form.
                This invention relates to bioenhanced formulations comprising eprosartan or eprosartan mesylate in the amorphous form, a process for its production, compositions containing the compound and methods of using the compound to block angiotensin II receptors and to treat hypertension, congestive heart failure and renal failure.
                """,
                "expected_output": [
                    "eprosartan mesylate",
                    "hypertension",
                    "congestive heart failure",
                    "renal failure",
                ],
            },
            {
                "text": """
                Pharmaceutical composition in particular for preventing and treating mucositis induced by radiotherapy or chemotherapy.
                The invention concerns a pharmaceutical composition designed to adhere to a mucous membrane in particular for preventing or treating radiotherapy-related and chemotherapy-related mucositis, induced by radiotherapy or combined radiochemotherapy, comprising an efficient amount of an antiradical compound mixed with a vehicle which is liquid at room temperature and gels at the mucous membrane temperature and capable of adhering to the mucous membrane by its gelled state.
                """,
                "expected_output": [
                    "radiotherapy",
                    "chemotherapy",
                    "radiotherapy relate",
                    "chemotherapy related mucositis",
                    "radiotherapy",
                    "combined radiochemotherapy",
                    "antiradical compound",
                ],
            },
        ]

        for condition in test_conditions:
            text = condition["text"]
            expected_output = condition["expected_output"]

            result = self.tagger.extract_strings([text], link=False)[0]
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
