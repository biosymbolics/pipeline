CONTROL_TERMS = [
    "placebo",
    "sham",
    "best supportive care",
    "standard",
    "usual care",
    "treatment as usual",
    "care as usual",
    "sugar pill",
    "routine",  # e.g. "routine care"
    "comparator",
    "best practice",
    "no treatment",
    "saline solution",
    "conventional",
    "aspirin",
    "control",
    "tablet dosage form",
    "laboratory biomarker analysis",
    "drug vehicle",
    "pharmacological study",
    "normal saline",
    "therapeutic procedure",
    "quality.?of.?life",
    "questionnaire",
]

DOSE_TERMS = [
    "dose",
    "doses",
    "dosing",
    "dosage",
    "dosages",
    "mad",
    "sad",
    "pharmacokinetic",
    "pharmacokinetics",
    "pharmacodynamics",
    "titration",
    "titrating",
]

WEEK = 7
MONTH = 30
YEAR = 365
DAY = 1
HOUR = 1 / 24
MINUTE = HOUR / 60
SECOND = MINUTE / 60

TIME_IN_DAYS: dict = {
    "second": SECOND,
    "minute": MINUTE,
    "hour": HOUR,
    "day": DAY,
    "week": WEEK,
    "month": MONTH,
    "year": YEAR,
}

TIME_UNITS: dict = {
    "second": ["seconds?", "s", "secs?"],
    "minute": ["minutes?", "mins?"],
    "hour": ["hours?", "hrs?", "hs?"],
    "day": ["days?", "ds?"],
    "week": ["weeks?", "wks?", "ws?"],
    "month": ["months?", "mons?", "mths?"],
    "year": ["years?", "yrs?", "ys?"],
}


DIGIT_MAP: dict = {
    "first": 1,
    "one": 1,
    "second": 2,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
