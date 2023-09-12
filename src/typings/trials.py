from datetime import date, datetime
from enum import Enum
from typing import TypeGuard, TypedDict, cast


class TrialStatus(Enum):
    PRE_ENROLLMENT = 1  # Active, not recruiting, Not yet recruiting
    ENROLLMENT = 2  # Recruiting, Enrolling by invitation
    WITHDRAWN = 3  # Withdrawn
    SUSPENDED = 4  # Suspended
    TERMINATED = 5  # Terminated
    COMPLETED = 6  # Completed
    UNKNOWN = 7  # Unknown status
    NA = 8  # Not Applicable

    @classmethod
    def _missing_(cls, value):
        """
        Example:
        TrialStatus("Active, not recruiting") -> TrialStatus.PRE_ENROLLMENT
        """
        status_term_map = {
            "Active, not recruiting": cls.PRE_ENROLLMENT,
            "Not yet recruiting": cls.PRE_ENROLLMENT,
            "Recruiting": cls.ENROLLMENT,
            "Enrolling by invitation": cls.ENROLLMENT,
            "Withdrawn": cls.WITHDRAWN,
            "Suspended": cls.SUSPENDED,
            "Terminated": cls.TERMINATED,
            "Completed": cls.COMPLETED,
            "Not Applicable": cls.NA,
        }
        if value in status_term_map:
            return status_term_map[value]
        return cls.UNKNOWN


class BaseTrial(TypedDict):
    """
    Base trial info

    TODO:
    - conditions -> disesases ?
    - interventions -> compounds ?
    """

    nct_id: str
    # blinding: str # TODO
    conditions: list[str]
    # design: str # parallel, crossover, etc
    # dropout
    end_date: date
    enrollment: int
    interventions: list[str]
    # intervention_design: str # TODO (whatever it is called - non-inferiority, superiority; active vs placebo comparator)
    last_updated_date: date
    phase: str
    # primary_endpoint: str # TODO (non-trivial)
    # randomization: str # TODO
    sponsor: str
    start_date: date
    # sponsor_type: str # TODO
    title: str
    # termination_reason: str


class TrialRecord(BaseTrial):
    status: str


class TrialSummary(BaseTrial):
    """
    Patent trial info
    """

    duration: int  # in days
    status: TrialStatus


def is_trial_record(trial: dict) -> TypeGuard[TrialRecord]:
    """
    Check if dict is a trial record
    """
    return (
        "nct_id" in trial
        and "end_date" in trial
        and "start_date" in trial
        and "last_updated_date" in trial
        and "status" in trial
        and "title" in trial
        and "phase" in trial
        and "conditions" in trial
        and "enrollment" in trial
        and "interventions" in trial
        and "sponsor" in trial
    )


def __calc_duration(start_date: date | None, end_date: date | None) -> int:
    """
    Calculate duration in days
    """
    if not start_date:
        return -1
    if not end_date:
        return (datetime.today().date() - start_date).days

    return (end_date - start_date).days


def get_trial_summary(trial: dict) -> TrialSummary:
    """
    Get trial summary from db record

    - formats start and end date
    - calculates duration

    TODO:
    - format phase
    - sponsor_type
    - termination_reason
    - primary endpoint
    """
    if not is_trial_record(trial):
        raise ValueError("Invalid trial record")

    status = TrialStatus(trial["status"])
    return cast(
        TrialSummary,
        {
            **trial,
            "duartion": __calc_duration(trial["start_date"], trial["end_date"]),
            "status": status,
        },
    )
