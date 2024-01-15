from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import Any, Sequence
from pydash import compact, flatten, uniq
from prisma.enums import TrialPhase, TrialStatus

from typings import ScoredRegulatoryApproval, ScoredPatent, ScoredTrial
from typings.documents.trials import TrialStatusGroup, get_trial_status_parent

from .core import Dataclass
from .documents.patents import MAX_PATENT_LIFE, AvailabilityLikelihood


@dataclass(frozen=True)
class Entity(Dataclass):
    activity: list[int]
    approval_count: int
    children: list["Entity"]
    enrollment: int
    id: str
    maybe_available_count: int
    name: str
    patent_count: int
    percent_stopped: float
    most_recent_patent: ScoredPatent | None
    most_recent_trial: ScoredTrial | None
    owners: list[str]
    trial_count: int

    def __init__(
        self,
        id: str,
        name: str,
        children: list["Entity"],
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        trials: list[ScoredTrial],
    ):
        object.__setattr__(self, "id", id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "children", children)
        object.__setattr__(
            self, "activity", self.get_activity(patents, regulatory_approvals, trials)
        )
        object.__setattr__(self, "approval_count", len(regulatory_approvals))
        object.__setattr__(self, "enrollment", self.get_enrollment(trials))
        object.__setattr__(self, "patent_count", len(patents))
        object.__setattr__(
            self, "most_recent_patent", self.get_most_recent_patent(patents)
        )
        object.__setattr__(
            self, "most_recent_trial", self.get_most_recent_trial(trials)
        )
        object.__setattr__(self, "trial_count", len(trials))
        object.__setattr__(
            self, "maybe_available_count", self.get_maybe_available_count(patents)
        )
        object.__setattr__(self, "owners", self.get_owners(patents, trials))
        object.__setattr__(self, "percent_stopped", self.get_percent_stopped(trials))

    @property
    def child_count(self):
        return len(self.children)

    def get_activity(
        self,
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        trials: list[ScoredTrial],
    ) -> list[int]:
        """
        Simple line chart of activity over time

        - patent filing priority dates
        - trial operational dates (start through end)
        - approval dates
        """
        dates = (
            [p.priority_date.year for p in patents]
            + flatten(
                [
                    list(range(t.start_date.year, t.end_date.year))
                    for t in trials
                    if t.end_date is not None and t.start_date is not None
                ]
            )
            + [a.approval_date for a in regulatory_approvals]
        )
        return [
            dates.count(y)
            for y in range(date.today().year - MAX_PATENT_LIFE, date.today().year + 1)
        ]

    def get_enrollment(self, trials) -> int:
        return sum([t.enrollment for t in trials if t.enrollment is not None]) or 0

    @property
    def investment_level(self) -> str:
        if self.enrollment > 5000:
            return "very high"
        if self.enrollment > 1000:
            return "high"
        if self.enrollment > 500:
            return "medium"
        return "low"

    @property
    def is_approved(self) -> bool:
        return self.approval_count > 0

    def get_most_recent_patent(
        self, patents: Sequence[ScoredPatent]
    ) -> ScoredPatent | None:
        if len(patents) == 0:
            return None
        patents = sorted(patents, key=lambda x: x.priority_date)
        return patents[-1]

    @property
    def last_priority_year(self) -> int | None:
        if not self.most_recent_patent:
            return None
        return self.most_recent_patent.priority_date.year

    def get_most_recent_trial(
        self, trials: Sequence[ScoredTrial]
    ) -> ScoredTrial | None:
        if len(trials) == 0:
            return None
        trials = sorted(trials, key=lambda x: x.last_updated_date)
        return trials[-1]

    @property
    def last_status(self) -> TrialStatus | str:
        if not self.most_recent_trial:
            return "??"
        return TrialStatus(self.most_recent_trial.status)

    @property
    def last_trial_updated_year(self) -> int | None:
        if not self.most_recent_trial:
            return None
        return self.most_recent_trial.last_updated_date.year

    @property
    def last_updated(self) -> int | None:
        p_updated = self.last_priority_year
        t_updated = self.last_trial_updated_year

        if not p_updated and not t_updated:
            return None

        return max(compact([p_updated, t_updated]))

    @property
    def max_phase(self) -> TrialPhase | str:
        if self.approval_count > 0:
            return TrialPhase.APPROVED
        if not self.most_recent_trial:
            return TrialPhase.PRECLINICAL
        return TrialPhase(self.most_recent_trial.phase)

    @property
    def record_count(self) -> int:
        return self.patent_count + self.approval_count + self.trial_count

    def get_maybe_available_count(self, patents: Sequence[ScoredPatent]) -> int:
        return [
            p.availability_likelihood
            in [
                AvailabilityLikelihood.POSSIBLE,
                AvailabilityLikelihood.LIKELY,
                "POSSIBLE",
                "LIKELY",
            ]
            for p in patents
        ].count(True)

    def get_owners(
        self, patents: Sequence[ScoredPatent], trials: Sequence[ScoredTrial]
    ) -> list[str]:
        return uniq(
            compact(
                flatten([a.canonical_name for p in patents for a in p.assignees or []])
                + [t.sponsor.canonical_name for t in trials if t.sponsor is not None]
            )
        )

    @property
    def owner_count(self) -> int:
        return len(self.owners)

    def get_percent_stopped(self, trials: list[ScoredTrial]) -> float:
        if self.trial_count == 0:
            return 0.0
        trial_statuses = [get_trial_status_parent(t.status) for t in trials]
        return trial_statuses.count(TrialStatusGroup.STOPPED) / len(trial_statuses)

    def serialize(self) -> dict[str, Any]:
        """
        Custom serialization for entity
        TODO: have trials/patents/approvals be pulled individually, maybe use Prisma
        """
        o = super().serialize()
        return {
            **o,
            "children": [c.serialize() for c in self.children],
        }

    # TODO - average dropout rate
    # TODO - average trial duration
    # TODO - chart with more info
