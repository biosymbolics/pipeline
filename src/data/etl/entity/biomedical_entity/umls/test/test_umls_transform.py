import unittest
from prisma.enums import OntologyLevel


from data.etl.entity.biomedical_entity.umls.umls_transform import (
    UmlsInfo,
    UmlsAncestorTransformer,
)


class TestUmlsUtils(unittest.TestCase):
    def test_choose_best_ancestor_from_type(self):
        test_cases = [
            {
                "description": "select target type ancestor for intervention type",
                "record": {
                    "level": OntologyLevel.SUBINSTANCE,
                    "id": "SELF_ID",
                    "type_ids": ["T121"],  # "Pharmacologic Substance"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "T121_ANCESTOR",
                            "type_ids": ["T121"],  # "Pharmacologic Substance"
                        },
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "TARGET_ANCESTOR",
                            "type_ids": ["T116"],  # "Amino Acid, Peptide, or Protein"
                        },
                    ]
                ),
                "expected": "TARGET_ANCESTOR",
            },
            {
                "description": "select less optimal target type ancestor for intervention type",
                "record": {
                    "level": OntologyLevel.SUBINSTANCE,
                    "id": "SELF_ID",
                    "type_ids": ["T121"],  # "Pharmacologic Substance"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "TARGET_ANCESTOR",
                            "type_ids": ["T121"],  # "Pharmacologic Substance"
                        },
                    ]
                ),
                "expected": "TARGET_ANCESTOR",
            },
            {
                "description": "select indication type ancestor for indication type",
                "record": {
                    "level": OntologyLevel.SUBINSTANCE,
                    "id": "SELF_ID",
                    "type_ids": ["T047"],  # "Disease or Syndrome"
                },
                "ancestors": tuple(
                    [
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C555555",
                            "type_ids": ["T116"],  # "Amino Acid, Peptide, or Protein"
                        },
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "C444444",
                            "type_ids": ["T047"],  # "Disease or Syndrome"
                        },
                    ]
                ),
                "expected": "C444444",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = UmlsAncestorTransformer.choose_best_ancestor(
                UmlsInfo(**test["record"], count=0, name=test["record"]["id"]),
                tuple(
                    [UmlsInfo(**a, count=0, name=a["id"]) for a in test["ancestors"]]
                ),
            )
            self.assertEqual(result, expected_output)

    def test_choose_best_ancestor_from_level(self):
        test_cases = [
            {
                "description": "skips over NA ancestors",
                "record": {
                    "level": OntologyLevel.SUBINSTANCE,
                    "id": "C1415265|C4721408",
                    "canonical_name": "gpr84 antagonist",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.NA, "id": "bb8"},
                        {"level": OntologyLevel.NA, "id": "cc7"},
                        {
                            "level": OntologyLevel.INSTANCE,
                            "id": "FIRST_INSTANCE_ANCESTOR",
                            "canonical_name": "gpr84",
                        },
                        {"level": OntologyLevel.L1_CATEGORY, "id": "dd6"},
                    ]
                ),
                "expected": "FIRST_INSTANCE_ANCESTOR",
            },
            {
                "description": "finds L1_CATEGORY ancestor",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "SELF_ANCESTOR",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "SAME_LEVEL_ANCESTOR"},
                        {
                            "level": OntologyLevel.L1_CATEGORY,
                            "id": "L1_CATEGORY_ANCESTOR",
                        },
                        {
                            "level": OntologyLevel.L2_CATEGORY,
                            "id": "L2_CATEGORY_ANCESTOR",
                        },
                    ]
                ),
                "expected": "L1_CATEGORY_ANCESTOR",
            },
            {
                "description": "Takes self as ancestor if already at level",
                "record": {
                    "level": OntologyLevel.L2_CATEGORY,
                    "id": "ALREADY_L2_CATEGORY",
                },
                "ancestors": tuple(
                    [
                        {"level": OntologyLevel.INSTANCE, "id": "INSTANCE_ANCESTOR"},
                        {
                            "level": OntologyLevel.L1_CATEGORY,
                            "id": "L1_CATEGORY_ANCESTOR",
                        },
                        {
                            "level": OntologyLevel.L2_CATEGORY,
                            "id": "L2_CATEGORY_ANCESTOR",
                        },
                    ]
                ),
                "expected": "ALREADY_L2_CATEGORY",
            },
            {
                "description": "Takes self as INSTANCE ancestor if no ancestors",
                "record": {
                    "level": OntologyLevel.INSTANCE,
                    "id": "SELF_ANCESTOR",
                    "canonical_name": "GPR84 gene",
                },
                "ancestors": tuple([]),
                "expected": "SELF_ANCESTOR",
            },
        ]

        for test in test_cases:
            expected_output = test["expected"]

            result = UmlsAncestorTransformer.choose_best_ancestor(
                UmlsInfo(
                    id=test["record"]["id"],
                    name="athing",
                    count=0,
                    level=test["record"]["level"],
                    type_ids=[],
                ),
                tuple(
                    [UmlsInfo(**a, count=0, name=a["id"]) for a in test["ancestors"]]
                ),
            )
            self.assertEqual(result, expected_output)
