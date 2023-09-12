from datetime import date
from typing import TypeGuard, TypedDict, cast


class BaseTrial(TypedDict):
    nct_id: str
    # blinding: str # TODO
    conditions: list[str]
    # design: str # parallel, crossover, etc
    enrollment: int
    interventions: list[str]
    # intervention_design: str # TODO (whatever it is called - non-inferiority, superiority; active vs placebo comparator)
    last_updated_date: date
    phase: str
    # primary_endpoint: str # TODO (non-trivial)
    # randomization: str # TODO
    sponsor: str
    status: str
    title: str
    # termination_reason: str


class TrialRecord(BaseTrial):
    end_date: str
    start_date: str


class TrialSummary(BaseTrial):
    """
    Patent trial info
    """

    duration: int  # in days
    end_date: date
    start_date: date


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


def get_trial_summary(trial: dict) -> TrialSummary:
    """
    Get trial summary from db record

    - formats start and end date
    - calculates duration
    """
    if not is_trial_record(trial):
        raise ValueError("Invalid trial record")

    start = date.fromisoformat(trial["start_date"])
    end = date.fromisoformat(trial["end_date"])
    return cast(
        TrialSummary,
        {**trial, "start_date": start, "end_date": end, "duartion": (end - start).days},
    )
