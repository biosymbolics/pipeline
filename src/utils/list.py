"""
Utils for lists/arrays
"""

from typing import Any, Iterable, Mapping, Sequence, TypeGuard, TypeVar, cast
import numpy as np
from pydash import compact, merge_with, uniq
import polars as pl

T = TypeVar("T")
BATCH_SIZE = 1000


def diff_lists(list_one: Sequence[T], list_two: Sequence[T]) -> list[T]:
    """
    Returns the items present in list_one but missing in list_two

    Args:
        list_one (list): list to compare
        list_two (list): list to compare
    """
    set_two = set(list_two)
    dropped = [x for x in list_one if x not in set_two]
    return dropped


def dedup(a_list: Sequence[T] | Iterable[T]) -> list[T]:
    """
    Returns a list with duplicate (and falsey) values removed

    Args:
        a_list (Sequence): list to deduplicate
    """
    return compact(list(set(a_list)))


def has_intersection(
    list_a: Sequence[T] | set[T], list_b: Sequence[T] | set[T]
) -> bool:
    """
    Returns True if list_a and list_b have at least one element in common

    Args:
        list_a (list): list to compare
        list_b (list): list to compare
    """
    return len(set(list_a).intersection(set(list_b))) > 0


def contains(list_a: Sequence[T] | set[T], list_b: Sequence[T] | set[T]) -> bool:
    """
    Returns True if list_a contains all elements in list_b

    Args:
        list_a (list): list to compare
        list_b (list): list to compare
    """
    return len(set(list_a).intersection(set(list_b))) == len(set(list_b))


def batch(items: Sequence[T], batch_size: int = BATCH_SIZE) -> list[list[T]]:
    """
    Turns a list into a list of lists of size `batch_size`

    Args:
        items (list): list to batch
        batch_size (int, optional): batch size. Defaults to BATCH_SIZE.
    """
    if batch_size == -1:
        return [list(items)]
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


BT = TypeVar("BT", bound=Mapping)


def batch_dict(data_dict: BT, batch_size: int = BATCH_SIZE) -> list[BT]:
    """
    Turns a dict of lists into a list of dicts of lists of size `batch_size`

    Args:
        data_dict (dict): dict to batch
        batch_size (int, optional): batch size. Defaults to BATCH_SIZE.
    """
    return cast(
        list[BT],
        [
            {k: v[i : i + batch_size] for k, v in data_dict.items()}
            for i in range(0, len(next(iter(data_dict.values()))), batch_size)
        ],
    )


def is_sequence(obj: object) -> bool:
    """
    Returns True if obj is a sequence (list, tuple, etc.)

    Args:
        obj (object): object to check
    """
    return not isinstance(obj, str) and isinstance(
        obj, (Sequence, list, tuple, pl.Series, np.ndarray)
    )


def is_tuple_list(x: Any) -> TypeGuard[list[tuple]]:
    """
    Simple typeguard for list of tuples
    """
    if is_sequence(x) and len(x) > 0 and all([isinstance(e, tuple) for e in x]):
        return True

    return False


def uniq_compact(array: Iterable[T | None]) -> list[T]:
    """
    Compact and deduplicate an array
    """
    return uniq(compact(array))


MT = TypeVar("MT")


def merge_nested(a: MT, *sources: MT) -> MT:
    """
    Merge two nested structures (dicts, lists, etc.)
    Lists are concatenated, dicts are merged. For everything else, a is returned.
    """

    def handle_merge(a: MT, b: MT):
        if isinstance(a, dict) and isinstance(b, dict):
            return merge_with(a, b, handle_merge)
        elif isinstance(a, list) and isinstance(b, list):
            return a + b
        else:
            return a

    return merge_with(a, *sources, handle_merge)
