"""
Patent types
"""
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Union

from typings.core import Dataclass


AutocompleteMode = Literal["id", "term"]
AutocompleteResult = TypedDict("AutocompleteResult", {"id": str, "label": str})


RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]


@dataclass(frozen=True)
class DocumentReportRecord(Dataclass):
    count: int
    x: str
    y: int | str | None = None


@dataclass(frozen=True)
class DocumentReport(Dataclass):
    x: str
    y: str | None
    data: list[DocumentReportRecord] | None
