from typing import Any, TypeGuard, Union, List, Dict

JsonSerializable = Union[
    Dict[str, "JsonSerializable"], List["JsonSerializable"], str, int, float, bool, None
]

Primitive = bool | str | int | float | None


def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """
    Checks if an object is a list of strings

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of strings
    """
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)
