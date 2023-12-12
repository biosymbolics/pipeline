"""
Utils related to arguments.
"""

import hashlib
import json


def parse_bool(value: bool | str | None) -> bool:
    if value is None:
        return False
    return json.loads(str(value).lower())


def make_hashable(obj):
    """
    Convert input to a hashable & comparable format
    """
    return hashlib.sha1(str(obj).encode()).hexdigest()
