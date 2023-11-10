"""
Utils for lists/arrays
"""


from typing import Mapping, Sequence, TypeVar, cast
import numpy as np
from pydash import compact
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


def dedup(a_list: Sequence[T]) -> list[T]:
    """
    Returns a list with duplicate (and falsey) values removed

    Args:
        a_list (Sequence): list to deduplicate
    """
    return compact(list(set(a_list)))


def has_intersection(list_a: Sequence[T] | set[T], list_b: Sequence[T]) -> bool:
    """
    Returns True if list_a and list_b have at least one element in common

    Args:
        list_a (list): list to compare
        list_b (list): list to compare
    """
    return len(set(list_a).intersection(set(list_b))) > 0


def batch(items: Sequence[T], batch_size: int = BATCH_SIZE) -> Sequence[Sequence[T]]:
    """
    Turns a list into a list of lists of size `batch_size`

    Args:
        items (list): list to batch
        batch_size (int, optional): batch size. Defaults to BATCH_SIZE.
    """
    if batch_size == -1:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


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
