from datetime import date
from typing import TypedDict


class TrialSummary(TypedDict):
    """
    Patent trial info
    """

    nct_id: str
    # blinding: str # TODO
    conditions: list[str]
    # design: str # parallel, crossover, etc
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
    status: str
    title: str
    # termination_reason: str
