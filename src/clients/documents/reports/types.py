from dataclasses import dataclass
from typing import Callable, Literal

from typings.core import Dataclass


@dataclass(frozen=True)
class DimensionInfo:
    is_entity: bool = False
    transform: Callable[[str], str] = lambda x: x


AggregationFunc = Literal["count"]
AggregationField = str


@dataclass(frozen=True)
class Aggregation:
    field: AggregationField
    func: AggregationFunc


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


CartesianDimension = Literal["x", "y"]
