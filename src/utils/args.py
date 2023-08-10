"""
Utils related to arguments.
"""

import hashlib


def make_hashable(obj):
    """
    Convert input to a hashable & comparable format
    """
    return hashlib.sha1(str(obj).encode()).hexdigest()
