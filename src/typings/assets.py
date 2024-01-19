from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence
from pydash import compact, count_by, flatten, group_by, uniq
from prisma.enums import TrialPhase, TrialStatus

from typings import ScoredRegulatoryApproval, ScoredPatent, ScoredTrial
from typings.documents.trials import TrialStatusGroup, get_trial_status_parent

from .core import Dataclass, EntityBase
from .documents.patents import MAX_PATENT_LIFE, AvailabilityLikelihood

OWNERS_LIMIT = 10


@dataclass(frozen=True)
class AssetActivity(Dataclass):
    # ids
    patents: list[str]
    regulatory_approvals: list[str]
    trials: list[str]
    year: int


class Asset(EntityBase):
    activity: list[int]
    detailed_activity: list[AssetActivity]
    average_trial_dropout: float
    average_trial_duration: int
    average_trial_enrollment: int
    children: list["Asset"]
    id: str
    maybe_available_count: int
    name: str
    owners: list[str]
    patent_count: int
    percent_trials_stopped: float
    most_recent_patent: ScoredPatent | None
    most_recent_trial: ScoredTrial | None
    regulatory_approval_count: int
    trial_count: int
    total_trial_enrollment: int

    @classmethod
    def load(
        cls,
        most_recent_patent: dict | None,
        most_recent_trial: dict | None,
        children: list[dict],
        **kwargs: Any
    ) -> "Asset":
        """
        Load from storage
        """
        asset = Asset(
            most_recent_patent=ScoredPatent(**most_recent_patent)
            if most_recent_patent
            else None,
            most_recent_trial=ScoredTrial(**most_recent_trial)
            if most_recent_trial
            else None,
            children=[Asset.load(**c) for c in children],
            **kwargs
        )
        return asset

    @classmethod
    def create(
        cls,
        id: str,
        name: str,
        children: list["Asset"],
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        trials: list[ScoredTrial],
    ) -> "Asset":
        return Asset(
            id=id,
            name=name,
            activity=cls.get_activity(patents, regulatory_approvals, trials),
            average_trial_dropout=cls.get_average_trial_dropout(trials),
            average_trial_duration=cls.get_average_trial_duration(trials),
            average_trial_enrollment=cls.get_average_trial_enrollment(trials),
            children=children,
            detailed_activity=cls.get_detailed_activity(
                patents, regulatory_approvals, trials
            ),
            maybe_available_count=cls.get_maybe_available_count(patents),
            most_recent_patent=cls.get_most_recent_patent(patents),
            most_recent_trial=cls.get_most_recent_trial(trials),
            owners=cls.get_owners(patents, trials),
            patent_count=len(patents),
            percent_trials_stopped=cls.get_percent_trials_stopped(trials),
            regulatory_approval_count=len(regulatory_approvals),
            total_trial_enrollment=cls.get_total_trial_enrollment(trials),
            trial_count=len(trials),
        )

    @property
    def child_count(self):
        return len(self.children)

    @property
    def investment_level(self) -> str:
        if self.total_trial_enrollment > 5000:
            return "very high"
        if self.total_trial_enrollment > 1000:
            return "high"
        if self.total_trial_enrollment > 500:
            return "medium"
        return "low"

    @property
    def is_approved(self) -> bool:
        return self.regulatory_approval_count > 0

    @property
    def last_priority_year(self) -> int | None:
        if not self.most_recent_patent:
            return None
        return self.most_recent_patent.priority_date.year

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
        if self.regulatory_approval_count > 0:
            return TrialPhase.APPROVED
        if not self.most_recent_trial:
            return TrialPhase.PRECLINICAL
        return TrialPhase(self.most_recent_trial.phase)

    @property
    def owner_count(self) -> int:
        return len(self.owners)

    @property
    def record_count(self) -> int:
        return self.patent_count + self.regulatory_approval_count + self.trial_count

    @classmethod
    def get_activity(
        cls,
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

    @classmethod
    def get_detailed_activity(
        cls,
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        trials: list[ScoredTrial],
    ) -> list[AssetActivity]:
        """
        Detailed activity over time

        - patent filing priority dates
        - trial *start dates* (todo: operational dates)
        - approval dates
        """
        patent_map = dict(group_by(patents, lambda p: p.priority_date.year))
        regulatory_approval_map = dict(
            group_by(regulatory_approvals, lambda a: a.approval_date.year)
        )
        trial_map = dict(
            group_by(
                trials,
                lambda t: t.start_date.year
                if t.start_date is not None
                else t.last_updated_date.year,
            )
        )

        return [
            AssetActivity(
                year=y,
                patents=[p.id for p in patent_map.get(y, [])],
                regulatory_approvals=[a.id for a in regulatory_approval_map.get(y, [])],
                trials=[t.id for t in trial_map.get(y, [])],
            )
            for y in range(date.today().year - MAX_PATENT_LIFE, date.today().year + 1)
        ]

    @classmethod
    def get_average_trial_enrollment(cls, trials) -> int:
        enrollments = [t.enrollment for t in trials if t.enrollment is not None]

        if len(enrollments) == 0:
            return 0
        return round(sum(enrollments) / len(enrollments))

    @classmethod
    def get_total_trial_enrollment(cls, trials) -> int:
        """
        Used as proxy for level of investment
        """
        enrollments = [t.enrollment for t in trials if t.enrollment is not None]

        if len(enrollments) == 0:
            return 0
        return sum(enrollments)

    @classmethod
    def get_most_recent_patent(
        cls, patents: Sequence[ScoredPatent]
    ) -> ScoredPatent | None:
        if len(patents) == 0:
            return None
        patents = sorted(patents, key=lambda x: x.priority_date)
        return patents[-1]

    @classmethod
    def get_most_recent_trial(cls, trials: Sequence[ScoredTrial]) -> ScoredTrial | None:
        if len(trials) == 0:
            return None
        trials = sorted(trials, key=lambda x: x.last_updated_date)
        return trials[-1]

    @classmethod
    def get_average_trial_dropout(cls, trials: list[ScoredTrial]) -> float:
        enroll_drop = [
            (t.enrollment, t.dropout_count)
            for t in trials
            if t.enrollment is not None
            and t.enrollment > 0
            and t.dropout_count is not None
            and t.dropout_count < (t.enrollment or 0)
        ]

        if len(enroll_drop) == 0:
            return 0.0

        return sum([d[1] for d in enroll_drop]) / sum([d[0] for d in enroll_drop])

    @classmethod
    def get_average_trial_duration(cls, trials: list[ScoredTrial]) -> int:
        durations = [trial.duration for trial in trials if trial.duration is not None]

        if len(durations) == 0:
            return 0

        return round(sum(durations) / len(durations))

    @classmethod
    def get_maybe_available_count(cls, patents: Sequence[ScoredPatent]) -> int:
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

    @classmethod
    def get_owners(
        cls, patents: Sequence[ScoredPatent], trials: Sequence[ScoredTrial]
    ) -> list[str]:
        # count and sort owners
        sorted_owners: list[tuple[str, int]] = sorted(
            count_by(
                compact(
                    flatten(
                        [a.canonical_name for p in patents for a in p.assignees or []]
                    )
                    + [
                        t.sponsor.canonical_name
                        for t in trials
                        if t.sponsor is not None
                    ]
                )
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # return top N owners
        return [o[0] for o in sorted_owners[0:OWNERS_LIMIT]]

    @classmethod
    def get_percent_trials_stopped(cls, trials: list[ScoredTrial]) -> float:
        if len(trials) == 0:
            return 0.0
        trial_statuses = [get_trial_status_parent(t.status) for t in trials]
        return trial_statuses.count(TrialStatusGroup.STOPPED) / len(trial_statuses)

    def serialize(self) -> dict[str, Any]:
        """
        Custom serialization for entity
        TODO: have trials/patents/approvals be pulled individually, maybe use Prisma
        """
        return {
            **super().serialize(),
            "children": [c.serialize() for c in self.children],
        }
