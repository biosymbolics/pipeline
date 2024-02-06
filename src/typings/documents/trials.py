import random
from prisma.partials import TrialDto
from prisma.enums import TerminationReason, TrialStatus

from typings.core import EntityBase
from utils.classes import ByDefinitionOrderEnum


class ScoredTrial(TrialDto, EntityBase):
    @property
    def dropout_percent(self) -> float | None:
        if self.enrollment is None or self.dropout_count is None:
            return None
        return self.dropout_count / self.enrollment

    @property
    def reformulation_score(self) -> float:
        """
        Score for reformulation potential

        **FAKE**!!
        """
        if (
            self.termination_reason is None
            or self.termination_reason == TerminationReason.NA
            or self.termination_reason == "NA"
        ):
            return 0.0

        return random.betavariate(2, 8)


class TrialStatusGroup(ByDefinitionOrderEnum):
    ONGOING = "ONGOING"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


def get_trial_status_parent(ts: TrialStatus) -> TrialStatusGroup | str:
    if ts in (TrialStatus.PRE_ENROLLMENT, TrialStatus.ENROLLMENT):
        return TrialStatusGroup.ONGOING
    if ts in (TrialStatus.WITHDRAWN, TrialStatus.SUSPENDED, TrialStatus.TERMINATED):
        return TrialStatusGroup.STOPPED
    if ts == TrialStatus.COMPLETED:
        return TrialStatusGroup.COMPLETED
    return TrialStatusGroup.UNKNOWN
