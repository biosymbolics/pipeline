"""
Patent types
"""
from typing_extensions import NotRequired
from typing import Literal, TypeGuard, TypedDict, Union
import typing

TermField = Literal["terms", "rollup_terms", "rollup_categories"]

AutocompleteTerm = TypedDict("AutocompleteTerm", {"id": str, "label": str})
TermResult = TypedDict("TermResult", {"term": str, "count": int})

RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]

PatentsReportRecord = TypedDict(
    "PatentsReportRecord",
    {"count": int, "x": str, "y": NotRequired[int | str]},
)
PatentsReport = TypedDict(
    "PatentsReport",
    {"x": str, "y": str | None, "data": list[PatentsReportRecord] | None},
)

QueryType = Literal["AND", "OR"]


def get_query_type(value: str | None) -> QueryType:
    """
    Get QueryType from string

    Args:
        value (str): value to check
    """
    if value is None:
        return "AND"
    if isinstance(value, str) and value in typing.get_args(QueryType):
        return value  # type: ignore
    raise ValueError(f"Invalid query type: {value}")


def is_relevancy_threshold(value: Union[str, tuple]) -> TypeGuard[RelevancyThreshold]:
    """
    Checks if a value is a RelevancyThreshold literal

    Args:
        value (Union[str, tuple]): value to check
    """
    return isinstance(value, str) and value in typing.get_args(RelevancyThreshold)
