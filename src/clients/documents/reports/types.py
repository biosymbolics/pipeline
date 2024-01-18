from dataclasses import dataclass
from typing import Callable, Literal


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
