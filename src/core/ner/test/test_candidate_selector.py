import unittest
import pytest

from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity

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
    # {
    #     "description": "keep unknown term if not enough terms of the right type match",
    #     "text": ["AABBCCxyz1 modulator"],
    #     "expected": [
    #         {
    #             "id": "aabbccxyz1|modulator",  # C0005525
    #             "name": "aabbccxyz1 Modulator",
    #         }
    #     ],
    # },
    # {
    #     "description": "shrink composite down to single match if only 1 term of the right type",
    #     "text": ["Maleimides Groups"],
    #     "expected": [
    #         {
    #             "id": "C0024579",
    #             "name": "Maleimides",
    #         }
    #     ],
    # },
    {
        "description": "apply canonical naming rules if single match composite",
        "text": ["LEUCINE-RICH REPEAT KINASE 2"],
        "expected": [
            {
                "id": "C2697910",
                "name": "LRRK2 Protein",
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
        "text": ["1,55"],
        "expected": [
            {
                "id": "",
                "name": "1,55",
            }
        ],
    },
    {
        "description": "is this right?",  # "test no match (should return None)",
        "text": ["4-pyrimidinediamine disodium"],
        "expected": [
            {
                "id": "C4742859",
                "name": "pyrimidine-4,6-diamine",  # "4-pyrimidinediamine disodium",
            }
        ],
    },
    {
        "description": "secondary match (optimization)",
        "text": ["gliflozin sodium-glucose cotransport 2 inhibitor"],
        "expected": [
            {
                "id": "C3273807",
                "name": "Sodium-Glucose Transporter 2 Inhibitors",
            }
        ],
    },
    {
        "description": "considering bigram match sufficient",  # TODO
        "text": ["blah blah inhibitor of SGLT2"],
        "expected": [
            {
                "id": "C3665047|blah",  # should be C3665047
                "name": "blah SGLT2 Inhibitors",
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
                "id": "C0005525|C1505133|C1564997",
                "name": "SLC5A1 SGLT2 Modulator",
            }
        ],
    },
    {
        "description": "composite match test",
        "text": ["c-aryl glucoside sglt2 inhibitor"],
        "expected": [
            {
                "id": "C0017765|C1505133|C1999216",
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
    {
        "description": "should match htr7",
        "text": ["5-ht7 receptor antagonists"],
        "expected": [
            {
                "id": "C1415816|C4721408",
                "name": "HTR7 Antagonist",
            }
        ],
    },
]

SEMANTIC_COMPOSITE_TEST_CASES = [
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
        "description": "composite match test with name override (SGLT2 > SLC5A2)",
        # "sglt1/sglt2 modulator" but ner splits on "/"
        "text": ["sglt1 sglt2 modulator"],
        "expected": [
            {
                "id": "C0005525|C0248805|C1565154",
                "name": "SGLT1 Protein SGLT2 Protein Modulator",
            }
        ],
    },
    {
        "description": "composite match test",
        "text": ["c-aryl glucoside sglt2 inhibitor"],
        "expected": [
            {
                "id": "C3273807",
                "name": "Sodium-Glucose Transporter 2 Inhibitors",
            }
        ],
    },
    {
        "description": "avoid gene match from common word",
        # avoid C1539188 / DNAAF6 ("TWISTER")
        "text": ["twist driver"],
        "expected": [
            {
                "id": "C3856475",
                "name": "Twist Drills",
            }
        ],
    },
]


@pytest.mark.skip(reason="Too slow to include in CI")
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
                        result.canonical_entity.__dict__[field],
                        test["expected"][0][field],
                    )


# TODO: add non-semantic composite logic here.
# @pytest.mark.skip(reason="Too slow to include in CI")
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

            print("Actual", result)
            print("Expected", test["expected"][0])

            if result is None or result.canonical_entity is None:
                self.assertEqual(None, test["expected"][0])
            elif test["expected"][0] is None:
                self.assertEqual(result.canonical_entity, None)
            else:
                for field in fields_to_test:
                    self.assertEqual(
                        result.canonical_entity.__dict__[field],
                        test["expected"][0][field],
                    )
        # self.assertEqual(True, False)


class TestCompositeTypeSelection(unittest.TestCase):
    def test_CanonicalEntity_type(self):
        test_cases = [
            {
                "description": "basic test",
                "ent": {
                    "id": "C0002716|C2916840|C3470309",
                    "name": "amyloid fam222a her inhibitors",
                    "types": ["T116", "T123", "T044", "T028"],
                },
                "expected": "BIOLOGIC",  # or mechanism
            },
            {
                "description": "skip over unknown TUI",
                "ent": {
                    "id": "CXXXYYYZZZ|C00XXXX",
                    "name": "some T116 thing",
                    "types": ["TUNKNOWN", "T116"],
                },
                "expected": "BIOLOGIC",
            },
            {
                "description": "handle BiomedicalEntityType",
                "ent": {
                    "id": "CXXXYYYZZZ",
                    "name": "some ent",
                    "types": ["MECHANISM"],
                },
                "expected": "MECHANISM",
            },
            {
                "description": "prefer BiomedicalEntityType",
                "ent": {
                    "id": "CXXXYYYZZZ",
                    "name": "some ent",
                    "types": ["T116", "MECHANISM"],
                },
                "expected": "MECHANISM",
            },
        ]

        for test in test_cases:
            e = CanonicalEntity(**test["ent"])
            self.assertEqual(e.type, test["expected"])
