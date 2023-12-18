from dataclasses import dataclass, field
from typing import Sequence

from .core import Dataclass
from .patents import Patent
from .trials import Trial


@dataclass(frozen=True)
class Entity(Dataclass):
    name: str
    patents: list[Patent]
    trials: list[Trial]

    @property
    def patent_count(self) -> int:
        return len(self.patents)

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def record_count(self) -> int:
        return self.trial_count + self.patent_count
