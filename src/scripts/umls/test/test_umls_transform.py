import unittest

from scripts.umls.transform import UmlsTransformer
from typings.umls import OntologyLevel


class TestTrialUtils(unittest.TestCase):
    def test_find_level_ancestor(self):
        test_cases = [
            {
                "description": "finds distant instance ancestor",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C1415265|C4721408",
                    "canonical_name": "gpr84 antagonist",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "bb8"},
                        {"level": OntologyLevel.INSTANCE, "id": "cc7"},
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C1415265",
                            "canonical_name": "gpr84",
                        },
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C1415265",
            },
            {
                "description": "takes self if no ancestors",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "VERY SPECIFIC THING",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "VERY SPECIFIC THING",
            },
            {
                "description": "finds L1_CATEGORY ancestor",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "VERY SPECIFIC THING",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.L1_CATEGORY],
                "expected": "dd6",
            },
            {
                "description": "finds L2_CATEGORY ancestor",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "VERY SPECIFIC THING",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.L2_CATEGORY],
                "expected": "ee5",
            },
            {
                "description": "Takes self as ancestor if already at level",
                "record": {
                    "level": OntologyLevel.L2_CATEGORY,
                    "id": "ALREADY_L2_CATEGORY",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.L2_CATEGORY],
                "expected": "ALREADY_L2_CATEGORY",
            },
            {
                "description": "Takes self as INSTANCE ancestor even if no ancestors",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C1415265",
                    "canonical_name": "GPR84 gene",
                },
                "ancestors": tuple([]),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C1415265",
            },
            {
                "description": "returns '' if no matching ancestors above INSTANCE",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C1415265",
                    "canonical_name": "GPR84 gene",
                },
                "ancestors": tuple([]),
                "levels": [OntologyLevel.L1_CATEGORY],
                "expected": "",
            },
            {
                "description": "does not go to lower level",
                "record": {
                    "level": OntologyLevel.L2_CATEGORY,
                    "id": "anl2cat",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.L1_CATEGORY],
                "expected": "anl2cat",
            },
            {
                "description": "multiple",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "anl2cat",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "levels": [OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY],
                "expected": "ee5",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = UmlsTransformer.find_level_ancestor(
                test["record"], test["levels"], test["ancestors"]
            )
            self.assertEqual(result, expected_output)
