"""
Miscellaneous utilities
"""
from collections import namedtuple
from typing import Any, NamedTuple


def dict_to_named_tuple(my_dict: dict[str, Any]) -> NamedTuple:
    """
    Convert a dict to a named tuple

    Args:
        my_dict (dict): dict to convert
    """
    MyTuple = namedtuple("MyTuple", list(my_dict.keys()))  # type: ignore # https://github.com/python/mypy/issues/848
    my_tuple = MyTuple(**my_dict)
    return my_tuple
