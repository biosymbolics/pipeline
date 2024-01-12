import unittest
import pytest

from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer

COMMON_COMPOSITE_TEST_CASES = [
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
                "name": "aabbcc Modulator",
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
        "expected": [
            {
                "id": "",
                "name": "-((6-oxo-1,6-dihydropyridazin-4-yl)methyl)piperazine",
            }
        ],
    },
    {
        "description": "no match for short partial terms",
        "text": ["1,3"],
        "expected": [
            {
                "id": "",
                "name": "1,3",
            }
        ],
    },
    {
        "description": "test no match (should return None)",
        "text": ["4-pyrimidinediamine disodium"],
        "expected": [
            {
                "id": "",
                "name": "4-pyrimidinediamine disodium",
            }
        ],
    },
    {
        "description": "secondary match (optimization)",
        "text": ["gliflozin sodium-glucose cotransport 2 inhibitor"],
        "expected": [
            {
                "id": "C1153347|C1999216|C3273807",  # TODO: should optimize to "Sodium-Glucose Transporter 2 Inhibitors"
                "name": "SGLT2 Inhibitor symporter activity Inhibitor",
            }
        ],
    },
    {
        "description": "considering bigram match sufficient",
        "text": ["blah blah inhibitor of SGLT2"],
        "expected": [
            {
                "id": "C3273807",
                "name": "Sodium-Glucose Transporter 2 Inhibitors",
            }
        ],
    },
]

NON_SEMANTIC_COMPOSITE_TEST_CASES = [
    {
        "description": "composite match test with name override (SGLT2 > SLC5A2)",
        "text": ["sglt1 sglt2 modulator"],
        "expected": [
            {
                "id": "C0005525|C1420200|C1420201",
                "name": "SLC5A1 SGLT2 Modulator",
            }
        ],
    },
    {
        "description": "composite match test",
        "text": ["c-aryl glucoside sglt2 inhibitor"],
        "expected": [
            {
                "id": "C0017765|C1420201|C1999216",
                "name": "Glucosides SGLT2 Inhibitor",
            }
        ],
    },
    {
        "description": "avoid gene match from common word",
        "text": ["twist driver"],
        "expected": [
            {
                "id": "C0533325",  # TODO?
                "name": "TWIST1",  # "Musculoskeletal torsion River driver",  # TODO
            }
        ],  # avoid C1539188 / DNAAF6 ("TWISTER")
    },
]

SEMANTIC_COMPOSITE_TEST_CASES = [
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
        "description": "composite match test",
        "text": ["c-aryl glucoside sglt2 inhibitor"],
        "expected": [
            {
                "id": "C0610842|C3890005|C5670121|c-aryl",
                "name": "c-aryl Glucosides SGLT2 Inhibitor",  # TODO: add to "is_partial" logic to semantic candidate selector
            }
        ],
    },
    {
        "description": "avoid gene match from common word",
        "text": ["twist driver"],
        "expected": [
            {
                "id": "",  # TODO?
                "name": "Musculoskeletal torsion River driver",  # TODO
            }
        ],  # avoid C1539188 / DNAAF6 ("TWISTER")
    },
]


class TestCompositeCandidateSelector(unittest.TestCase):
    """
    Note: test initialization is slow because of the UMLS cache
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_selector: CandidateSelectorType = "CompositeCandidateSelector"

    def setUp(self):
        self.normalizer = TermNormalizer(candidate_selector=self.candidate_selector)

    def test_composite_candidate_selector(self):
        test_cases = [*COMMON_COMPOSITE_TEST_CASES, *NON_SEMANTIC_COMPOSITE_TEST_CASES]

        fields_to_test = ["id", "name"]

        for test in test_cases:
            text = test["text"]

            result = self.normalizer.normalize_strings(text)[0]
            print("RESULT", result)

            if result is None or result.canonical_entity is None:
                self.assertEqual(None, test["expected"][0])
            elif test["expected"][0] is None:
                self.assertEqual(result.canonical_entity, None)
            else:
                for field in fields_to_test:
                    self.assertEqual(
                        result.canonical_entity._asdict()[field],
                        test["expected"][0][field],
                    )


# TODO: add non-semantic composite logic here.
@pytest.mark.skip(reason="Too slow to include in CI")
class TestCompositeSemanticCandidateSelector(unittest.TestCase):
    """
    Note: test initialization is slow because of the UMLS cache
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_selector: CandidateSelectorType = (
            "CompositeSemanticCandidateSelector"
        )

    def setUp(self):
        self.normalizer = TermNormalizer(candidate_selector=self.candidate_selector)

    def test_composite_candidate_semantic_selector(self):
        test_cases = [*COMMON_COMPOSITE_TEST_CASES, *SEMANTIC_COMPOSITE_TEST_CASES]

        fields_to_test = ["id", "name"]

        for test in test_cases:
            text = test["text"]
            result = self.normalizer.normalize_strings(text)[0]

            if result is None or result.canonical_entity is None:
                self.assertEqual(None, test["expected"][0])
            elif test["expected"][0] is None:
                self.assertEqual(result.canonical_entity, None)
            else:
                for field in fields_to_test:
                    self.assertEqual(
                        result.canonical_entity._asdict()[field],
                        test["expected"][0][field],
                    )
