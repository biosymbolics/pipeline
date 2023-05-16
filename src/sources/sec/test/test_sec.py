"""
SEC module tests
"""
import unittest
from datetime import datetime

from sources.sec.product_pipeline import get_pipeline_by_ticker


class TestSec(unittest.TestCase):
    def test_fetch_quarterly_reports(self):
        """
        Tests for quarterly reports
        """
        test_cases = [
            {
                "ticker": "PFE",
                "start": datetime(2022, 1, 1),
                "expected": 10,
            },
        ]

        for test in test_cases:
            result = get_pipeline_by_ticker(test["ticker"], test["start"])
            self.assertEqual(len(result), test["expected"])
