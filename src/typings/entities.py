from dataclasses import dataclass
from datetime import date
from typing import Any
from pydash import compact, flatten, uniq

from typings.approvals import RegulatoryApproval

from .core import Dataclass
from .patents import MAX_PATENT_LIFE, AvailabilityLikelihood, Patent
from .trials import Trial, TrialPhase, TrialStatus


@dataclass(frozen=True)
class Entity(Dataclass):
    name: str
    approvals: list[RegulatoryApproval]
    patents: list[Patent]
    trials: list[Trial]

    @property
    def activity(self) -> list[int]:
        """
        Simple line chart of activity over time
        """
        dates = (
            [p.priority_date.year for p in self.patents]
            + [t.last_updated_date.year for t in self.trials]
            + flatten([[ad.year for ad in a.approval_dates] for a in self.approvals])
        )
        return [
            dates.count(y)
            for y in range(date.today().year - MAX_PATENT_LIFE, date.today().year + 1)
        ]

    @property
    def approval_count(self) -> int:
        return sum(a.count for a in self.approvals)

    @property
    def is_approved(self) -> bool:
        return self.approval_count > 0

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
            return "??"
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
            # TODO: why isn't p.availability_likelihood the enum?
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
            flatten([p.assignees for p in self.patents])
            + [t.normalized_sponsor for t in self.trials]
        )

    @property
    def owner_count(self) -> int:
        return len(self.owners)

    def serialize(self) -> dict[str, Any]:
        """
        Custom serialization for entity
        TODO: have trials/patents/approvals be pulled individually, maybe use Prisma
        """
        trials = [t.serialize() for t in self.trials]
        patents = [p.serialize() for p in self.patents]
        approvals = [a.serialize() for a in self.approvals]

        o = super().serialize()
        return {**o, "approvals": approvals, "patents": patents, "trials": trials}

    # TODO - average dropout rate
    # TODO - average trial duration
    # TODO - chart with more info
