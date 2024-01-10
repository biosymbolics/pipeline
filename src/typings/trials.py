from prisma.models import Trial
from prisma.enums import TrialStatus

from typings.core import EntityBase
from utils.classes import ByDefinitionOrderEnum


class ScoredTrial(Trial, EntityBase):
    @property
    def dropout_percent(self) -> float | None:
        if self.enrollment is None or self.dropout_count is None:
            return None
        return self.dropout_count / self.enrollment


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
