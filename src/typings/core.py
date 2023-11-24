from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, TypeGuard, Union, List, Dict

JsonSerializable = Union[
    Dict[str, "JsonSerializable"], List["JsonSerializable"], str, int, float, bool, None
]

Primitive = bool | str | int | float | None


@dataclass(frozen=True)
class Dataclass:
    def __getitem__(self, item):
        return getattr(self, item)

    def get(self, item, default=None):
        return getattr(self, item, default)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _asdict(self) -> dict[str, Any]:
        return asdict(self)

    def keys(self):
        return self._asdict().keys()

    def values(self):
        return self._asdict().values()

    def items(self):
        return self._asdict().items()

    def replace(self, **kwargs):
        return replace(self, **kwargs)


def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """
    Checks if an object is a list of strings

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of strings
    """
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)


def is_string_list_list(obj: Any) -> TypeGuard[list[list[str]]]:
    """
    Checks if an object is a list of string lists

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of string lists
    """
    return (
        isinstance(obj, list)
        and all(isinstance(x, list) for x in obj)
        and all(isinstance(x, str) for x in obj[0])
    )


def is_dict_list(obj: Any) -> TypeGuard[list[dict[str, Any]]]:
    """
    Checks if an object is a list of dicts

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of dicts
    """
    return isinstance(obj, list) and all(isinstance(x, dict) for x in obj)
