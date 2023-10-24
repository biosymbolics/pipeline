import unittest

from scripts.ctgov.utils import extract_timeframe


class TestTrialUtils(unittest.TestCase):
    def test_extract_timefraome(self):
        test_cases = [
            {
                "time_frame_desc": "From Day 0 up to Year 2",
                "expected": 365 * 2,
            },
            {
                "time_frame_desc": "Change between baseline and follow-up at 26-weeks",
                "expected": 26 * 7,
            },
            {
                "time_frame_desc": "Day 1 of Part A up to Week 24 of Part B",
                "expected": 24 * 7,
            },
            {
                "time_frame_desc": "Date of enrollment to study completion (Date of 5 year follow-up, withdrawal or death).",
                "expected": 365 * 5,
            },
            {
                "time_frame_desc": "at 12-month visit",
                "expected": 12 * 30,
            },
            {
                "time_frame_desc": "Week 0, Week 26 and Week 52",
                "expected": 52 * 7,
            },
            {
                "time_frame_desc": "Weeks 0-26 and Weeks 0-52",
                "expected": 52 * 7,
            },
            {
                "time_frame_desc": "Efficacy Expansion: Baseline up to Day 1072",
                "expected": 1072,
            },
            {
                "time_frame_desc": "Duration of hospital stay (average 3.4 days)",
                "expected": 3,
            },
            # {
            #     "time_frame_desc": "Cycle 1 Day -3: Predose, 1, 2, 4, 6, 8, 10, 24, 48 and 72 hours (hr) postdose; Cycle 1 Day 28: Predose, 1, 2, 4, 6, 8, 10 and 24 hr postdose (Cycle 1 = 32 days)",
            #     "expected": 28,  # TODO?
            # },
            # {
            #     "time_frame_desc": "At Day 15, 29, 43, 57, 71, 85, 99, 127 and 169",
            #     "expected": 169,
            # },
        ]

        for test in test_cases:
            time_frame_desc = test["time_frame_desc"]
            expected_output = test["expected"]

            result = extract_timeframe(time_frame_desc)
            print("Actual", result, "expected", expected_output)
            self.assertEqual(result, expected_output)
