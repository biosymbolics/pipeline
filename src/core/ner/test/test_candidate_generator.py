import unittest

from core.ner.linker.candidate_generator import CompositeCandidateGenerator


class TestCandidateGenerator(unittest.TestCase):
    """
    Note: test initialization is slow because of the UMLS cache
    """

    def setUp(self):
        self.candidate_generator = CompositeCandidateGenerator(min_similarity=0.85)

    def test_candidate_generator(self):
        test_cases = [
            {
                "description": "basic match test",
                "text": ["sodium glucose transporter 2 inhibitor"],
                "expected": [
                    {
                        "id": "C3273807",
                        "name": "Sodium-Glucose Transporter 2 Inhibitors",
                    }
                ],
            },
            {
                "description": "composite match test",
                "text": ["c-aryl glucoside sglt2 inhibitor"],
                "expected": [
                    {
                        "id": "C0017765|C3273807",
                        "name": "Glucosides SGLT2 Inhibitor",
                    }
                ],
            },
            {
                "description": "composite match test with name override (SGLT2 > SLC5A2)",
                # "sglt1/sglt2 modulator" but ner splits on "/"
                "text": ["sglt1 sglt2 modulator"],
                "expected": [
                    {
                        "id": "C0005525|C1420201",
                        "name": "SGLT2 Modulator",
                    }
                ],
            },
            {
                "description": "drop extra garbage term if 2+ terms match",
                "text": ["AABBCC sglt2 modulator"],
                "expected": [
                    {
                        "id": "C0005525|C1420201",
                        "name": "SGLT2 Modulator",
                    }
                ],
            },
            {
                "description": "keep unknown term if not enough terms of the right type match",
                "text": ["AABBCC modulator"],
                "expected": [
                    {
                        "id": "C0005525|aabbcc",
                        "name": "AABBCC Modulator",
                    }
                ],
            },
            {
                "description": "shrink composite down to single match if only 1 term of the right type",
                "text": ["Maleimides Groups"],
                "expected": [
                    {
                        "id": "C0024579",
                        "name": "Maleimides",
                    }
                ],
            },
            {
                "description": "apply canonical naming rules if single match composite",
                "text": ["LEUCINE-RICH REPEAT KINASE 2"],
                "expected": [
                    {
                        "id": "C1425650",
                        "name": "LRRK2",
                    }
                ],
            },
            {
                "description": "apply composite naming rules if 2+ match composite",
                "text": ["ROCO2 inhibitor"],  # ROCO2 == LRRK2
                "expected": [
                    {
                        "id": "C1425650|C1999216",
                        "name": "LRRK2 Inhibitor",  # versus "LRRK2 gene inhibitor"
                    }
                ],
            },
            {
                "description": "single match IUPAC name",
                "text": ["3'-azido-2',3'-dideoxyuridine"],
                "expected": [
                    {
                        "id": "C0046786",
                        "name": "3'-azido-2',3'-dideoxyuridine",
                    }
                ],
            },
            {
                "description": "no composite match for IUPAC name",
                "text": ["-((6-oxo-1,6-dihydropyridazin-4-yl)methyl)piperazine"],
                "expected": [None],
            },
            {
                "description": "no match for short partial terms",
                "text": ["1,3"],
                "expected": [None],
            },
            {
                "description": "test no match (should return None)",
                "text": ["4-pyrimidinediamine disodium"],
                "expected": [None],
            },
            {
                "description": "avoid gene match from common word",
                "text": ["twist driver"],
                "expected": [
                    {
                        "id": "C0040480|C0335471",
                        "name": "Musculoskeletal torsion River driver",  # TODO
                    }
                ],  # avoid C1539188 / DNAAF6 ("TWISTER")
            },
        ]

        fields_to_test = ["id", "name"]

        for test in test_cases:
            text = test["text"]

            result = self.candidate_generator(text)[0]
            print("RESULT", result)

            for field in fields_to_test:
                if test["expected"][0] is None:
                    self.assertEqual(result, None)
                else:
                    self.assertEqual(
                        result._asdict()[field], test["expected"][0][field]
                    )
