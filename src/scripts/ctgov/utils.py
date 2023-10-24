from pydash import compact, flatten, is_number
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


def extract_timeframe(timeframe_desc: str) -> int | None:
    """
    Extract outcome durations in days
    """
    unit_re = get_or_re(flatten(time_units.values()))
    digit_re = rf"(?:(?:[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+)|(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve))"
    digit_units_re = rf"{digit_re}+[ -]?{unit_re}"
    time_joiners = "(?:[-, ]+| to | and )"
    units_digit_re = rf"\b{unit_re}[ -]?{digit_re}+(?:{time_joiners}+{digit_re}+)*"
    timeframe_re = rf"(?:{digit_units_re}|{units_digit_re}|{digit_re})"

    timeframe_candidates = re.findall(
        timeframe_re, timeframe_desc, re.IGNORECASE | re.MULTILINE
    )

    def get_unit(time_desc: str) -> str | None:
        units = [
            k
            for k, v in time_units.items()
            if re.search(rf"\b{get_or_re(v)}\b", time_desc, re.IGNORECASE) is not None
        ]
        if len(units) == 0:
            return None

        if len(units) > 1:
            raise ValueError(f"Multiple units found: {units}")

        return units[0]

    def calc_time(time_desc: str) -> int | None:
        digits = re.findall(digit_re, time_desc)
        unit = get_unit(time_desc)

        if unit is None:
            return None

        v = [int(d) * time_in_days[unit] for d in digits if is_number(int(d))]
        if len(v) == 0:
            return None
        return round(max(v))

    times = compact([calc_time(candidate) for candidate in timeframe_candidates])

    return max(times) if len(times) > 0 else None


def extract_max_timeframe(timeframe_descs: list[str]) -> int | None:
    """
    Returns the largest timeframe in days
    """
    times = compact(
        [extract_timeframe(timeframe_desc) for timeframe_desc in timeframe_descs]
    )
    return max(times) if len(times) > 0 else None
