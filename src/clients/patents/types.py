"""
Patent types
"""
from typing import Literal, TypeGuard, TypedDict, Union
import typing

TermResult = TypedDict("TermResult", {"term": str, "count": int})

RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]


def is_relevancy_threshold(value: Union[str, tuple]) -> TypeGuard[RelevancyThreshold]:
    """
    Checks if a value is a RelevancyThreshold literal

    Args:
        value (Union[str, tuple]): value to check
    """
    return isinstance(value, str) and value in typing.get_args(RelevancyThreshold)
