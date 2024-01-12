from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class DimensionInfo:
    is_entity: bool = False
    transform: Callable[[str], str] = lambda x: x
