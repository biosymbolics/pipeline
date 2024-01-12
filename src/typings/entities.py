from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import Any
from pydash import compact, flatten, uniq
from prisma.enums import TrialPhase, TrialStatus

from typings import ScoredRegulatoryApproval, ScoredPatent, ScoredTrial
from typings.documents.trials import TrialStatusGroup, get_trial_status_parent

from .core import Dataclass
from .documents.patents import MAX_PATENT_LIFE, AvailabilityLikelihood


@dataclass(frozen=True)
class Entity(Dataclass):
    id: int
    name: str
    child_count: int
    children: list["Entity"]
    patents: list[ScoredPatent]
    regulatory_approvals: list[ScoredRegulatoryApproval]
    trials: list[ScoredTrial]

    @property
    def activity(self) -> list[int]:
        """
        Simple line chart of activity over time

        - patent filing priority dates
        - trial operational dates (start through end)
        - approval dates
        """
        dates = (
            [p.priority_date.year for p in self.patents]
            + flatten(
                [
                    list(range(t.start_date.year, t.end_date.year))
                    for t in self.trials
                    if t.end_date is not None and t.start_date is not None
                ]
            )
            + [a.approval_date for a in self.regulatory_approvals]
        )
        return [
            dates.count(y)
            for y in range(date.today().year - MAX_PATENT_LIFE, date.today().year + 1)
        ]

    @property
    def approval_count(self) -> int:
        return len(self.regulatory_approvals)

    @property
    def total_enrollment(self) -> int:
        return sum([t.enrollment for t in self.trials if t.enrollment is not None]) or 0

    @property
    def investment_level(self) -> str:
        if self.total_enrollment > 5000:
            return "very high"
        if self.total_enrollment > 1000:
            return "high"
        if self.total_enrollment > 500:
            return "medium"
        return "low"

    @property
    def is_approved(self) -> bool:
        return self.approval_count > 0

    @property
    def patent_count(self) -> int:
        return len(self.patents)

    @cached_property
    def most_recent_patent(self) -> ScoredPatent | None:
        if len(self.patents) == 0:
            return None
        patents = sorted(self.patents, key=lambda x: x.priority_date)
        return patents[-1]

    @property
    def last_priority_year(self) -> int | None:
        if not self.most_recent_patent:
            return None
        return self.most_recent_patent.priority_date.year

    @cached_property
    def most_recent_trial(self) -> ScoredTrial | None:
        if len(self.trials) == 0:
            return None
        trials = sorted(self.trials, key=lambda x: x.last_updated_date)
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
    def trial_count(self) -> int:
        return len(self.trials)

    @property
    def record_count(self) -> int:
        return self.trial_count + self.patent_count

    @property
    def maybe_available_count(self) -> int:
        return [
            p.availability_likelihood
            in [
                AvailabilityLikelihood.POSSIBLE,
                AvailabilityLikelihood.LIKELY,
                "POSSIBLE",
                "LIKELY",
            ]
            for p in self.patents
        ].count(True)

    @property
    def owners(self) -> list[str]:
        return uniq(
            compact(
                flatten(
                    [a.canonical_name for p in self.patents for a in p.assignees or []]
                )
                + [
                    t.sponsor.canonical_name
                    for t in self.trials
                    if t.sponsor is not None
                ]
            )
        )

    @property
    def owner_count(self) -> int:
        return len(self.owners)

    @property
    def percent_stopped(self) -> float:
        if self.trial_count == 0:
            return 0.0
        trial_statuses = [get_trial_status_parent(t.status) for t in self.trials]
        return trial_statuses.count(TrialStatusGroup.STOPPED) / len(trial_statuses)

    def serialize(self) -> dict[str, Any]:
        """
        Custom serialization for entity
        TODO: have trials/patents/approvals be pulled individually, maybe use Prisma
        """
        trials = [t.serialize() for t in self.trials]
        patents = [p.serialize() for p in self.patents]
        regulatory_approvals = [a.serialize() for a in self.regulatory_approvals]

        o = super().serialize()
        return {
            **o,
            "children": [c.serialize() for c in self.children],
            "regulatory_approvals": regulatory_approvals,
            "patents": patents,
            "trials": trials,
        }

    # TODO - average dropout rate
    # TODO - average trial duration
    # TODO - chart with more info
