"""
Utils for lists/arrays
"""


from typing import TypeVar


def diff_lists(list_one: list, list_two: list) -> list:
    """
    Returns the items present in list_one but missing in list_two
    """
    set_two = set(list_two)
    dropped = [x for x in list_one if x not in set_two]
    return dropped


T = TypeVar("T")


def dedup(a_list: list[T]) -> list[T]:
    """
    Returns a list with duplicates removed

    Args:
        a_list (list): list to deduplicate
    """
    return list(set(a_list))


def has_intersection(list_a: list[T], list_b: list[T]) -> bool:
    """
    Returns True if list_a and list_b have at least one element in common
    """
    return len(set(list_a).intersection(set(list_b))) > 0
