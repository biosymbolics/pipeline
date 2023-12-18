from dataclasses import dataclass
from typing import Sequence

from .core import Dataclass
from .patents import Patent
from .trials import Trial


@dataclass(frozen=True)
class Entity(Dataclass):
    name: str
    patents: Sequence[Patent]
    trials: Sequence[Trial]
