from typing import Any, TypeGuard, Union, List, Dict
from spacy.tokens import Doc as SpacyDoc

JsonSerializable = Union[
    Dict[str, "JsonSerializable"], List["JsonSerializable"], str, int, float, bool, None
]

Primitive = bool | str | int | float | None

Doc = SpacyDoc


def is_string_list(obj: Any) -> TypeGuard[list[str]]:
    """
    Checks if an object is a list of strings

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of strings
    """
    return isinstance(obj, list) and all(isinstance(x, str) for x in obj)


def is_doc_list(obj: Any) -> TypeGuard[list[SpacyDoc]]:
    """
    Checks if an object is a list of SpaCy docs

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of SpaCy docs
    """
    return isinstance(obj, list) and all(isinstance(x, SpacyDoc) for x in obj)
