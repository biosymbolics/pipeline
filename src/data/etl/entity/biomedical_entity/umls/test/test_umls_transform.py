import unittest

from typings.umls import OntologyLevel

from data.etl.entity.biomedical_entity.umls.transform import (
    UmlsInfo,
    UmlsAncestorTransformer,
)


class TestTrialUtils(unittest.TestCase):
    def test_choose_best_ancestor_from_type(self):
        test_cases = [
            {
                "description": "select target type ancestor for intervention type",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C123123123",
                    "type_ids": ["T121"],  # "Pharmacologic Substance"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C444444",
                            "type_ids": ["T121"],  # "Pharmacologic Substance"
                        },
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C555555",
                            "type_ids": ["T116"],  # "Amino Acid, Peptide, or Protein"
                        },
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C555555",
            },
            {
                "description": "select indication type ancestor for indication type",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C123123123",
                    "type_ids": ["T047"],  # "Disease or Syndrome"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C444444",
                            "type_ids": ["T047"],  # "Disease or Syndrome"
                        },
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C555555",
                            "type_ids": ["T116"],  # "Amino Acid, Peptide, or Protein"
                        },
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C444444",
            },
            {
                "description": "choose based on level of no type match (furthest LEVEL ancestor)",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "C123123123",
                    "type_ids": ["T121"],  # "Pharmacologic Substance"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C444444",
                            "type_ids": ["T121"],  # "Pharmacologic Substance"
                        },
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C555555",
                            "type_ids": ["T121"],  # "Pharmacologic Substance"
                        },
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C555555",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = UmlsAncestorTransformer.choose_best_ancestor(
                UmlsInfo(
                    id=test["record"]["id"],
                    level=test["record"]["level"],
                    type_ids=test["record"]["type_ids"],
                ),
                test["levels"],
                tuple([UmlsInfo(**a) for a in test["ancestors"]]),
            )
            self.assertEqual(result, expected_output)

    def test_choose_best_ancestor_from_level(self):
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
                "description": "use self if no ancestors at the desired level, and self is at or above desired level",
                "record": {
                    "level": OntologyLevel.L2_CATEGORY,
                    "id": "C2987634",
                    "canonical_name": "agonist",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.L2_CATEGORY, "id": "C0450442"},
                        {"level": OntologyLevel.L2_CATEGORY, "id": "C1254372"},
                    ]
                ),
                "levels": [OntologyLevel.INSTANCE],
                "expected": "C2987634",
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

            result = UmlsAncestorTransformer.choose_best_ancestor(
                UmlsInfo(
                    id=test["record"]["id"], level=test["record"]["level"], type_ids=[]
                ),
                test["levels"],
                tuple([UmlsInfo(**a) for a in test["ancestors"]]),
            )
            self.assertEqual(result, expected_output)
