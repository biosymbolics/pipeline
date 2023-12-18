from dataclasses import dataclass, field
from datetime import date
from typing import Sequence

from .core import Dataclass
from .patents import Patent
from .trials import Trial, TrialPhase, TrialStatus


@dataclass(frozen=True)
class Entity(Dataclass):
    name: str
    patents: list[Patent]
    trials: list[Trial]

    @property
    def activity(self) -> list[int]:
        """
        Simple line chart of activity over time
        """
        dates = [p.priority_date.year for p in self.patents] + [
            t.last_updated_date.year for t in self.trials
        ]

        return [
            dates.count(y) for y in range(date.today().year - 20, date.today().year)
        ]

    @property
    def patent_count(self) -> int:
        return len(self.patents)

    @property
    def most_recent_patent(self) -> Patent | None:
        if len(self.patents) == 0:
            return None
        patents = sorted(self.patents, key=lambda x: x.priority_date)
        return patents[-1]

    @property
    def last_priority_year(self) -> int | None:
        if not self.most_recent_patent:
            return None
        return self.most_recent_patent.priority_date.year

    @property
    def most_recent_trial(self) -> Trial | None:
        if len(self.trials) == 0:
            return None
        trials = sorted(self.trials, key=lambda x: x.last_updated_date)
        return trials[-1]

    @property
    def last_status(self) -> TrialStatus | None:
        if not self.most_recent_trial:
            return None
        return TrialStatus(self.most_recent_trial.status)

    @property
    def last_updated(self) -> date | None:
        if not self.most_recent_trial:
            return None
        return self.most_recent_trial.last_updated_date

    @property
    def max_phase(self) -> TrialPhase:
        return sorted(
            [TrialPhase(t.phase) for t in self.trials], key=lambda x: x._order
        )[-1]

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def record_count(self) -> int:
        return self.trial_count + self.patent_count
