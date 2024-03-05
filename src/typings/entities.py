from dataclasses import dataclass
from typing import Any, Sequence
from pydantic import Field, computed_field
from pydash import compact, count_by, flatten, group_by
from prisma.enums import TrialPhase, TrialStatus
import logging
from constants.documents import MAX_DATA_YEAR

from typings import ScoredRegulatoryApproval, ScoredPatent, ScoredTrial
from typings.documents.trials import (
    TRIAL_PHASE_ORDER,
    TrialStatusGroup,
    get_trial_status_parent,
)

from .core import Dataclass, ResultBase
from .documents.patents import AvailabilityLikelihood

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

OWNERS_LIMIT = 10


@dataclass(frozen=True)
class EntityActivity(Dataclass):
    # id lists are too big
    patents: int
    regulatory_approvals: int
    trials: int
    year: int


class Entity(ResultBase):
    activity: list[int]
    detailed_activity: list[EntityActivity]
    average_trial_dropout: float | None
    average_trial_duration: int | None
    average_trial_enrollment: int | None
    children: list["Entity"]
    id: str
    investment: int
    maybe_available_count: int
    maybe_available_ids: list[str]
    max_phase: TrialPhase
    name: str
    owners: list[str]
    patent_count: int
    patent_ids: list[str]
    patent_weight: int
    percent_trials_stopped: float | None
    most_recent_patent: ScoredPatent | None = Field(exclude=True)
    most_recent_trial: ScoredTrial | None = Field(exclude=True)
    regulatory_approval_count: int
    regulatory_approval_ids: list[str]
    total_completed_trial_enrollment: int | None
    traction: int
    trial_count: int
    trial_ids: list[str]
    total_trial_enrollment: int | None

    @classmethod
    def load(
        cls,
        most_recent_patent: dict | None,
        most_recent_trial: dict | None,
        children: list[dict],
        **kwargs: Any
    ) -> "Entity":
        """
        Load from storage
        """
        entity = Entity(
            most_recent_patent=(
                ScoredPatent(**most_recent_patent) if most_recent_patent else None
            ),
            most_recent_trial=(
                ScoredTrial(**most_recent_trial) if most_recent_trial else None
            ),
            children=[Entity.load(**c) for c in children],
            **kwargs
        )
        return entity

    @classmethod
    def create(
        cls,
        id: str,
        name: str,
        children: list["Entity"],
        end_year: int,
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        start_year: int,
        trials: list[ScoredTrial],
        is_child: bool = False,
    ) -> "Entity":
        maybe_available_ids = cls.get_maybe_available_ids(patents)
        return Entity(
            id=id,
            name=name,
            activity=(
                cls.get_activity(
                    start_year, end_year, patents, regulatory_approvals, trials
                )
            ),
            average_trial_dropout=cls.get_average_trial_dropout(trials),
            average_trial_duration=cls.get_average_trial_duration(trials),
            average_trial_enrollment=cls.get_average_trial_enrollment(trials),
            children=children,
            # detailed_activity gets large, so don't include it for children.
            detailed_activity=(
                cls.get_detailed_activity(
                    start_year, end_year, patents, regulatory_approvals, trials
                )
                if not is_child
                else []
            ),
            investment=sum(
                [p.investment for p in patents]
                + [a.investment for a in regulatory_approvals]
                + [t.investment for t in trials]
            ),
            maybe_available_count=len(maybe_available_ids),
            maybe_available_ids=maybe_available_ids,
            max_phase=cls.get_max_phase(trials, len(regulatory_approvals)),
            most_recent_patent=cls.get_most_recent_patent(patents),
            most_recent_trial=cls.get_most_recent_trial(trials),
            owners=cls.get_owners(patents, regulatory_approvals, trials),
            patent_count=len(patents),
            patent_ids=[p.id for p in patents],
            patent_weight=cls.get_patent_weight(patents),
            percent_trials_stopped=cls.get_percent_trials_stopped(trials),
            regulatory_approval_count=len(regulatory_approvals),
            regulatory_approval_ids=[a.id for a in regulatory_approvals],
            total_completed_trial_enrollment=cls.get_total_trial_enrollment(
                trials, [TrialStatus.COMPLETED]
            ),
            total_trial_enrollment=cls.get_total_trial_enrollment(trials),
            traction=sum(
                [p.traction for p in patents]
                + [a.traction for a in regulatory_approvals]
                + [t.traction for t in trials]
            ),
            trial_count=len(trials),
            trial_ids=[t.id for t in trials],
        )

    @computed_field  # type: ignore # https://github.com/python/mypy/issues/14461
    @property
    def child_count(self) -> int:
        return len(self.children)

    @classmethod
    def get_patent_weight(cls, patents: Sequence[ScoredPatent]) -> int:
        """
        Count all patents - not just the WO, but the country-specific and other variants
        (proxy for LOI in each patent)
        """
        return sum([len(p.other_ids) for p in patents])

    @computed_field  # type: ignore
    @property
    def investment_level(self) -> str:
        if self.investment > 5000:
            return "very high"
        if self.investment > 1000:
            return "high"
        if self.investment > 500:
            return "medium"

        return "low"

    @computed_field  # type: ignore
    @property
    def traction_level(self) -> str:
        if self.traction > 1000:
            return "very high"
        if self.traction > 500:
            return "high"
        if self.traction > 100:
            return "medium"

        return "low"

    @computed_field  # type: ignore
    @property
    def is_approved(self) -> bool:
        return self.regulatory_approval_count > 0

    @computed_field  # type: ignore
    @property
    def last_priority_year(self) -> int | None:
        if not self.most_recent_patent:
            return None
        return self.most_recent_patent.priority_date.year

    @computed_field  # type: ignore
    @property
    def last_status(self) -> TrialStatus | str:
        if not self.most_recent_trial:
            return "??"
        return TrialStatus(self.most_recent_trial.status)

    @computed_field  # type: ignore
    @property
    def last_trial_updated_year(self) -> int | None:
        if not self.most_recent_trial:
            return None
        return self.most_recent_trial.last_updated_date.year

    @computed_field  # type: ignore
    @property
    def last_updated(self) -> int | None:
        p_updated = self.last_priority_year
        t_updated = self.last_trial_updated_year

        if not p_updated and not t_updated:
            return None

        return max(compact([p_updated, t_updated]))

    @classmethod
    def get_max_phase(
        cls, trials: Sequence[ScoredTrial], regulatory_approval_count: int
    ) -> TrialPhase:
        if regulatory_approval_count > 0:
            return TrialPhase.APPROVED
        if len(trials) == 0:
            return TrialPhase.PRECLINICAL

        phases = sorted(
            [TrialPhase(t.phase) for t in trials if t.phase is not None],
            key=lambda x: TRIAL_PHASE_ORDER[x],
            reverse=True,
        )
        return phases[0]

    @computed_field  # type: ignore
    @property
    def owner_count(self) -> int:
        return len(self.owners)

    @computed_field  # type: ignore
    @property
    def record_count(self) -> int:
        return self.patent_count + self.regulatory_approval_count + self.trial_count

    @classmethod
    def get_activity(
        cls,
        start_year: int,
        end_year: int,
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
            + [a.approval_date.year for a in regulatory_approvals]
        )
        return [dates.count(y) for y in range(start_year, min(MAX_DATA_YEAR, end_year))]

    @classmethod
    def get_detailed_activity(
        cls,
        start_year: int,
        end_year: int,
        patents: list[ScoredPatent],
        regulatory_approvals: list[ScoredRegulatoryApproval],
        trials: list[ScoredTrial],
    ) -> list[EntityActivity]:
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
                lambda t: (
                    t.start_date.year
                    if t.start_date is not None
                    else t.last_updated_date.year
                ),
            )
        )

        return [
            EntityActivity(
                year=y,
                patents=len([p.id for p in patent_map.get(y, [])]),
                regulatory_approvals=len(
                    [a.id for a in regulatory_approval_map.get(y, [])]
                ),
                trials=len([t.id for t in trial_map.get(y, [])]),
            )
            for y in range(start_year, end_year)
        ]

    @classmethod
    def get_average_trial_enrollment(cls, trials) -> int | None:
        if len(trials) == 0:
            return None

        enrollments = [t.enrollment for t in trials if t.enrollment is not None]

        if len(enrollments) == 0:
            return 0
        return round(sum(enrollments) / len(enrollments))

    @classmethod
    def get_total_trial_enrollment(
        cls,
        trials: Sequence[ScoredTrial],
        statuses: Sequence[TrialStatus] | None = None,
    ) -> int | None:
        """
        Get total enrollment for trials.

        Args:
            trials (list[ScoredTrial]): list of trials
            statuses (list[TrialStatus], optional): filter by status. Defaults to None (all statuses)
        """
        if len(trials) == 0:
            return None

        enrollments = [
            t.enrollment
            for t in trials
            if t.enrollment is not None and (statuses is None or t.status in statuses)
        ]

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
    def get_average_trial_dropout(cls, trials: list[ScoredTrial]) -> float | None:
        enroll_drop = [
            (t.enrollment, t.dropout_count)
            for t in trials
            if t.enrollment is not None
            and t.enrollment > 0
            and t.dropout_count is not None
            and t.dropout_count < (t.enrollment or 0)
        ]

        if len(trials) == 0:
            return None

        if len(enroll_drop) == 0:
            return 0.0

        return sum([d[1] for d in enroll_drop]) / sum([d[0] for d in enroll_drop])

    @classmethod
    def get_average_trial_duration(cls, trials: list[ScoredTrial]) -> int | None:
        if len(trials) == 0:
            return None

        durations = [trial.duration for trial in trials if trial.duration is not None]

        if len(durations) == 0:
            return 0

        return round(sum(durations) / len(durations))

    @classmethod
    def get_maybe_available_ids(cls, patents: Sequence[ScoredPatent]) -> list[str]:
        return [
            p.id
            for p in patents
            if p.availability_likelihood
            in [
                AvailabilityLikelihood.POSSIBLE,
                AvailabilityLikelihood.LIKELY,
                "POSSIBLE",
                "LIKELY",
            ]
        ]

    @classmethod
    def get_owners(
        cls,
        patents: Sequence[ScoredPatent],
        regulatory_approvals: Sequence[ScoredRegulatoryApproval],
        trials: Sequence[ScoredTrial],
    ) -> list[str]:
        # count and sort owners
        sorted_owners: list[tuple[str, int]] = sorted(
            count_by(
                compact(
                    flatten(
                        [a.canonical_name for p in patents for a in p.assignees or []]
                    )
                    + [
                        a.applicant.canonical_name
                        for a in regulatory_approvals
                        if a.applicant is not None
                    ]
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
    def get_percent_trials_stopped(cls, trials: list[ScoredTrial]) -> float | None:
        if len(trials) == 0:
            return None
        trial_statuses = [get_trial_status_parent(t.status) for t in trials]
        return trial_statuses.count(TrialStatusGroup.STOPPED) / len(trial_statuses)

    def serialize(self) -> dict[str, Any]:
        """
        Custom serialization for entity
        """
        return {
            **super().serialize(),
            "children": [c.serialize() for c in self.children],
        }
