from pydash import flatten, is_number
import regex as re

from utils.re import get_or_re

WEEK = 7
MONTH = 30
YEAR = 365
DAY = 1
HOUR = 1 / 24
MINUTE = HOUR / 60
SECOND = MINUTE / 60

time_units: dict = {
    "second": ["seconds?", "s", "secs?"],
    "minute": ["minutes?", "mins?"],
    "hour": ["hours?", "hrs?", "hs?"],
    "day": ["days?", "ds?"],
    "week": ["weeks?", "wks?", "ws?"],
    "month": ["months?", "mons?", "mths?"],
    "year": ["years?", "yrs?", "ys?"],
}

time_in_days: dict = {
    "second": SECOND,
    "minute": MINUTE,
    "hour": HOUR,
    "day": DAY,
    "week": WEEK,
    "month": MONTH,
    "year": YEAR,
}


def extract_timeframe(timeframe_desc: str) -> int:
    """
    Calculate outcome durations in days
    """
    # choose the last
    unit_re = get_or_re(flatten(time_units.values()))
    digit_re = rf"(?:(?:[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+)|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))"
    digit_units_re = rf"{digit_re}+[ -]?{unit_re}"
    units_digit_re = rf"\b{unit_re}[ -]?{digit_re}+(?:-{digit_re}+)?"  # weeks 1-12

    timeframe_re = rf"(?:{digit_units_re}|{units_digit_re}|{digit_re})"

    timeframe_candidates = re.findall(
        timeframe_re, timeframe_desc, re.IGNORECASE | re.MULTILINE
    )

    print("PSS", timeframe_candidates)

    def calc_time(time: str) -> int:
        digits = re.findall(digit_re, time)
        units = [
            k
            for k, v in time_units.items()
            if re.search(rf"\b{get_or_re(v)}\b", time, re.IGNORECASE) is not None
        ]
        print(units)
        unit = units[0]
        v = [int(d) * time_in_days[unit] for d in digits if is_number(int(d))]
        print("timeframe_desc", time, unit, time_in_days[unit], digits, v)
        return round(max(v))

    times = [calc_time(candidate) for candidate in timeframe_candidates]
    return max(times)
