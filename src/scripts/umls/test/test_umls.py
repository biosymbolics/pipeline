import unittest

from scripts.umls.utils import find_level_ancestor
from typings.umls import OntologyLevel


class TestTrialUtils(unittest.TestCase):
    def test_find_level_ancestor(self):
        test_cases = [
            {
                "description": "finds distant instance ancestor",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "VERY SPECIFIC THING",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "aa9"},
                        {"level": OntologyLevel.INSTANCE, "id": "bb8"},
                        {"level": OntologyLevel.INSTANCE, "id": "cc7"},
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "ee5"},
                    ]
                ),
                "level": OntologyLevel.INSTANCE,
                "expected": "cc7",
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
                "level": OntologyLevel.INSTANCE,
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
                "level": OntologyLevel.L1_CATEGORY,
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
                "level": OntologyLevel.L2_CATEGORY,
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
                "level": OntologyLevel.L2_CATEGORY,
                "expected": "ALREADY_L2_CATEGORY",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = find_level_ancestor(
                test["record"], test["level"], test["ancestors"]
            )
            self.assertEqual(result, expected_output)
