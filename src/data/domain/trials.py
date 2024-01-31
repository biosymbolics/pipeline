from pydash import compact, flatten
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

digit_map: dict = {
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


def extract_timeframe(timeframe_desc: str | None) -> int | None:
    """
    Extract outcome durations in days
    """
    unit_re = get_or_re(flatten(time_units.values()))
    number_re = rf"(?:(?:[0-9]+|[0-9]+\.[0-9]+|[0-9]+,[0-9]+)|{get_or_re(list(digit_map.keys()))})"
    number_units_re = rf"{number_re}+[ -]?{unit_re}"
    time_joiners = "(?:[-, ±]+| to | and )"
    units_digit_re = rf"\b{unit_re}[ -]?{number_re}+(?:{time_joiners}+{number_re}+)*"
    timeframe_re = rf"(?:{number_units_re}|{units_digit_re}|{number_re})"

    if timeframe_desc is None:
        return None

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

    def get_number(d: str) -> int | None:
        if d in digit_map:
            return digit_map[d]

        try:
            return int(d)
        except ValueError:
            return None

    def calc_time(time_desc: str) -> int | None:
        number_strs = re.findall(number_re, time_desc, re.IGNORECASE)
        unit = get_unit(re.sub(get_or_re(number_strs), "", time_desc))

        if unit is None:
            return None

        numbers = compact([get_number(num_str.lower()) for num_str in number_strs])
        v = [int(n) * time_in_days[unit] for n in numbers]
        if len(v) == 0:
            return None
        return round(max(v))

    times = compact([calc_time(candidate) for candidate in timeframe_candidates])

    max_timeframe = max(times) if len(times) > 0 else None
    return max_timeframe


def extract_max_timeframe(timeframe_descs: list[str]) -> int | None:
    """
    Returns the largest timeframe in days
    """
    times = compact(
        [
            extract_timeframe(timeframe_desc)
            for timeframe_desc in compact(timeframe_descs or [])
        ]
    )
    return max(times) if len(times) > 0 else None
