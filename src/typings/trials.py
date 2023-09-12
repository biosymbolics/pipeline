from datetime import date, datetime
from enum import Enum
from typing import TypeGuard, TypedDict, cast
from core.ner.classifier import classify_by_keywords

from utils.list import dedup
from utils.re import expand_res


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


class TrialPhase(Enum):
    EARLY_PHASE_1 = 1  # Early Phase 1
    PHASE_1 = 2  # Phase 1
    PHASE_1_2 = 3  # Phase 1/Phase 2
    PHASE_2 = 4  # Phase 2
    PHASE_2_3 = 5  # Phase 2/Phase 3
    PHASE_3 = 6  # Phase 3
    PHASE_4 = 7  # Phase 4
    NA = 8  # Not Applicable
    UNKNOWN = 9  # Unknown status

    @classmethod
    def _missing_(cls, value):
        """
        Example:
        TrialStatus("Active, not recruiting") -> TrialStatus.PRE_ENROLLMENT
        """
        phase_term_map = {
            "Early Phase 1": cls.EARLY_PHASE_1,
            "Phase 1": cls.PHASE_1,
            "Phase 1/Phase 2": cls.PHASE_1_2,
            "Phase 2": cls.PHASE_2,
            "Phase 2/Phase 3": cls.PHASE_2_3,
            "Phase 3": cls.PHASE_3,
            "Phase 4": cls.PHASE_4,
            "Not Applicable": cls.NA,
        }
        if value in phase_term_map:
            return phase_term_map[value]
        return cls.UNKNOWN


class TerminationReason(Enum):
    FUTILITY = 1
    SAFETY = 2
    BUSINESS = 3
    FAILED_TO_ENROLL = 4
    LOSS_OF_FOLLOW_UP = 5
    INVESTIGATOR = 6
    FUNDING = 7
    COVID = 8
    OVERSIGHT = 9
    SUPPLY_CHAIN = 10
    PROTOCOL_REVISION = 11
    FEASIBILITY = 12
    NOT_SAFETY = 13
    LOGISTICS = 14
    NOT_EFFICACY = 15

    @classmethod
    def _missing_(cls, value):
        reason = classify_by_keywords(
            [value], cast(dict[str, list[str]], TERMINATION_KEYWORD_MAP), "OTHER"
        )
        return reason[0]


TERMINATION_KEYWORD_MAP = {
    TerminationReason.FUTILITY: [
        "futility",
        # "efficacy",
        "endpoints?",  # "failed to meet (?:primary )?endpoint", "no significant difference on (?:primary )?endpoint", "efficacy endpoints?", "efficacy endpoints"
        "lower success rates?",
        "interim analysis",
        "lack of effectiveness",
        "lack of efficacy",
        "rate of relapse",
        "lack of response",
        "lack of performance",
        "inadequate effect",
        "no survival benefit",
        "stopping rule",
    ],
    TerminationReason.SAFETY: [
        # "safety", # "not a safety issue"
        "toxicity",
        "adverse",
        "risk/benefit",
        "detrimental effect",
        "S?AEs?",
        "mortality",
        # "safety concerns?",
        "unacceptable morbidity",
        "side effects?",
    ],
    TerminationReason.BUSINESS: [
        "business",
        "company",
        "strategic",
        "sponsor(?:'s)? decision",
        "management",
        "stakeholders?",
        "(?:re)?prioritization",
    ],
    TerminationReason.FAILED_TO_ENROLL: [
        "accruals?",
        "enroll?(?:ment|ed)?",
        "inclusions?",
        "recruit(?:ment|ing)?s?",
        "lack of (?:eligibile )?(?:participants?|subjects?|patients?)",
        "lack of ",
    ],
    TerminationReason.LOSS_OF_FOLLOW_UP: ["lost to follow up"],
    TerminationReason.INVESTIGATOR: ["investigator", "PI"],
    TerminationReason.FUNDING: ["funding", "resources", "budget", "financial"],
    TerminationReason.COVID: ["covid", "coronavirus", "pandemic"],
    TerminationReason.OVERSIGHT: [
        "IRB",
        "ethics",
        "Institutional Review Board",
        "certification",
        "FDA Hold",
    ],
    TerminationReason.SUPPLY_CHAIN: [
        "supply",
        "unavalaible",
        "shortage",
        "manufacturing",
    ],
    TerminationReason.PROTOCOL_REVISION: ["revision", "change in (?:study )?protocol"],
    TerminationReason.FEASIBILITY: ["feasibility"],
    TerminationReason.NOT_SAFETY: [
        "not a safety issue",
        "not related to safety",
        "No Safety or Efficacy Concerns",
        "no safety concern",
    ],
    TerminationReason.NOT_EFFICACY: [
        "not related to efficacy",
        "No Safety or Efficacy Concerns",
    ],
    TerminationReason.LOGISTICS: ["logistics", "logistical"],
}


def get_termination_reason(why_stopped: str) -> str:
    return why_stopped


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
    - sponsor_type
    - termination_reason
    - primary endpoint
    """
    if not is_trial_record(trial):
        raise ValueError("Invalid trial record")

    return cast(
        TrialSummary,
        {
            **trial,
            "conditions": dedup(trial["conditions"]),
            "duartion": __calc_duration(trial["start_date"], trial["end_date"]),
            "phase": TrialPhase(trial["phase"]),
            "status": TrialStatus(trial["status"]),
        },
    )
