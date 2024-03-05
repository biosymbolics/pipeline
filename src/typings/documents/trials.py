import random
from prisma.partials import TrialDto
from prisma.enums import TerminationReason, TrialPhase, TrialStatus
from pydantic import computed_field

from typings.core import ResultBase
from utils.classes import ByDefinitionOrderEnum


class ScoredTrial(TrialDto, ResultBase):
    @computed_field  # type: ignore
    @property
    def dropout_percent(self) -> float | None:
        if (
            self.enrollment is None
            or self.enrollment == 0
            or self.dropout_count is None
        ):
            return None
        return self.dropout_count / self.enrollment

    @computed_field  # type: ignore
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


TRIAL_PHASE_ORDER: dict[TrialPhase, float] = {
    TrialPhase.PRECLINICAL: 0,
    TrialPhase.EARLY_PHASE_1: 0.5,
    TrialPhase.PHASE_1: 1,
    TrialPhase.PHASE_1_2: 1.5,
    TrialPhase.PHASE_2: 2,
    TrialPhase.PHASE_2_3: 2.5,
    TrialPhase.PHASE_3: 3,
    TrialPhase.APPROVED: 4,
    TrialPhase.PHASE_4: -1,
    TrialPhase.NA: -1,
    TrialPhase.UNKNOWN: -1,
}
