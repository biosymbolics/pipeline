import unittest

from core.ner.linker.candidate_generator import CompositeCandidateGenerator


class TestCandidateGenerator(unittest.TestCase):
    def setUp(self):
        self.candidate_generator = CompositeCandidateGenerator(min_similarity=0.85)

    def test_candidate_genreator(self):
        test_cases = [
            {
                "text": ["sodium glucose co-transporter 2 (sglt2) inhibitor"],
                "expected": [
                    {
                        "id": "C2983740",
                        "name": "Sodium-glucose co-transporter 2 inhibitor",
                        "type": "T123",
                    }
                ],
            },
        ]

        fields_to_test = ["id", "name", "type"]

        for test in test_cases:
            text = test["text"]

            result = self.candidate_generator(text)[0]

            for field in fields_to_test:
                self.assertEqual(result.__dict__[field], test["expected"][0][field])
