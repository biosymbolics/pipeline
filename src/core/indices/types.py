from typing import Any, TypeGuard, TypedDict

EntityObj = TypedDict("EntityObj", {"name": str, "details": str})


def is_entity_obj(obj: Any) -> TypeGuard[EntityObj]:
    """
    Check if object is an EntityObj

    Args:
        obj (Any): object to check
    """
    if not isinstance(obj, dict):
        return False
    return obj.get("name") is not None and obj.get("details") is not None
