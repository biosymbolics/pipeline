import unittest
from pydash import group_by

from constants.company import OTHER_OWNER_NAME

from nlp.clustering.clustering import (
    cluster_terms,
    _create_cluster_term_map,
)


class TestClustering(unittest.TestCase):
    def test_cluster_terms(self):
        test_conditions = [
            {
                "description": "basic clustering",
                "input": [
                    "tgf beta",
                    "tgf beta other",  # making sure this name is not chosen
                    "tgf beta",
                    "thing other",
                    "thing other",
                ],
                "expected": {
                    "tgf beta": "tgf beta",
                    "tgf beta other": "tgf beta",
                    "thing other": OTHER_OWNER_NAME,
                },
            },
        ]

        for condition in test_conditions:
            expected = condition["expected"]

            grouped = group_by(condition["input"], lambda x: x)
            counts = [len(v) for v in grouped.values()]
            terms = list(grouped.keys())
            input = dict(zip(terms, counts))

            result = cluster_terms(input)
            if result != expected:
                print(f"Actual: '{result}', expected: '{expected}'")

            self.assertEqual(result, expected)

    def test__create_cluster_term_map(self):
        test_conditions = [
            {
                "description": "selects the most common term",
                "terms": [
                    "canonical_name",
                    "non_canonical_name",
                    "canonical_name",
                ],
                "cluster_ids": [1, 1, 1],
                "expected": {
                    "non_canonical_name": "canonical_name",
                    "canonical_name": "canonical_name",
                },
            },
        ]

        for test in test_conditions:
            terms = test["terms"]
            cluster_ids = test["cluster_ids"]
            expected = test["expected"]

            result, other = _create_cluster_term_map(
                terms, [1] * len(terms), cluster_ids
            )
            if result != expected:
                print(f"Actual: '{result}', expected: '{expected}'")

            self.assertEqual(result, expected)
