"""
Patent types
"""
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard, TypedDict, Union
import typing

from typings.core import Dataclass
from utils.list import is_sequence


AutocompleteMode = Literal["id", "term"]
AutocompleteResult = TypedDict("AutocompleteResult", {"id": str, "label": str})
TermResult = TypedDict("TermResult", {"term": str, "count": int})


def is_term_result(obj: Any) -> TypeGuard[TermResult]:
    """
    Checks if an object is a TermResult

    Args:
        obj (Any): object to check
    """
    return isinstance(obj, dict) and "term" in obj and "count" in obj


def is_term_results(obj: Any) -> TypeGuard[list[TermResult]]:
    """
    Checks if an object is a list of TermResults

    Args:
        obj (Any): object to check
    """
    return is_sequence(obj) and all(is_term_result(x) for x in obj)


RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]


@dataclass(frozen=True)
class PatentsReportRecord(Dataclass):
    count: int
    x: str
    y: int | str | None = None


@dataclass(frozen=True)
class PatentsReport(Dataclass):
    x: str
    y: str | None
    data: list[PatentsReportRecord] | None


def is_relevancy_threshold(value: Union[str, tuple]) -> TypeGuard[RelevancyThreshold]:
    """
    Checks if a value is a RelevancyThreshold literal

    Args:
        value (Union[str, tuple]): value to check
    """
    return isinstance(value, str) and value in typing.get_args(RelevancyThreshold)


@dataclass(frozen=True)
class QueryPieces(Dataclass):
    fields: list[str]
    froms: list[str]
    wheres: list[str]
    params: list
