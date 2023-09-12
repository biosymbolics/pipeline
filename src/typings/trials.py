from datetime import date, datetime
from enum import IntEnum
from typing import TypeGuard, TypedDict, cast
from constants.company import LARGE_PHARMA_KEYWORDS

from core.ner.classifier import classify_string, create_lookup_map
from utils.list import dedup


class TrialRandomization(IntEnum):
    RANDOMIZED = "RANDOMIZED"
    NON_RANDOMIZED = "NON_RANDOMIZED"
    NA = "N/A"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        rand_term_map = {
            "Randomized": cls.RANDOMIZED,
            "Non-Randomized": cls.NON_RANDOMIZED,
            "N/A": cls.NA,
        }
        if value in rand_term_map:
            return rand_term_map[value]
        return cls.UNKNOWN


class TrialMasking(IntEnum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    QUADRUPLE = "QUADRUPLE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        masking_term_map = {
            "None (Open Label)": cls.NONE,
            "Single": cls.SINGLE,
            "Double": cls.DOUBLE,
            "Triple": cls.TRIPLE,
            "Quadruple": cls.QUADRUPLE,
        }
        if value in masking_term_map:
            return masking_term_map[value]
        return cls.UNKNOWN


class TrialPurpose(IntEnum):
    TREATMENT = "TREATMENT"
    PREVENTION = "PREVENTION"
    DIAGNOSTIC = "DIAGNOSTIC"
    BASIC_SCIENCE = "BASIC_SCIENCE"
    SUPPORTIVE_CARE = "SUPPORTIVE_CARE"
    DEVICE = "DEVICE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        purpose_term_map = {
            "Treatment": cls.TREATMENT,
            "Prevention": cls.PREVENTION,
            "Diagnostic": cls.DIAGNOSTIC,
            "Basic Science": cls.BASIC_SCIENCE,
            "Supportive Care": cls.SUPPORTIVE_CARE,
            "Device Feasibility": cls.DEVICE,
            "Educational/Counseling/Training": cls.OTHER,
            "Health Services Research": cls.OTHER,
            "Screening": cls.OTHER,
        }
        if value in purpose_term_map:
            return purpose_term_map[value]
        return cls.UNKNOWN


class TrialDesign(IntEnum):
    PARALLEL = "PARALLEL"
    CROSSOVER = "CROSSOVER"
    FACTORIAL = "FACTORIAL"
    SEQUENTIAL = "SEQUENTIAL"
    SINGLE_GROUP = "SINGLE_GROUP"
    NA = "N/A"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        design_term_map = {
            "Parallel Assignment": cls.PARALLEL,
            "Crossover Assignment": cls.CROSSOVER,
            "Factorial Assignment": cls.FACTORIAL,
            "Sequential Assignment": cls.SEQUENTIAL,
            "Single Group Assignment": cls.SINGLE_GROUP,
            "": cls.NA,
        }
        if value in design_term_map:
            return design_term_map[value]
        return cls.UNKNOWN


class SponsorType(IntEnum):
    UNIVERSITY = "UNIVERSITY"
    INDUSTRY_LARGE = "INDUSTRY_LARGE"  # big pharma
    INDUSTRY = "INDUSTRY"
    GOVERNMENTAL = "GOVERNMENTAL"
    HEALTH_SYSTEM = "HEALTH_SYSTEM"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value):
        reason = classify_string(value, SPONSOR_KEYWORD_MAP, cls.OTHER)  # type: ignore
        res = reason[0]
        return res


SPONSOR_KEYWORD_MAP = create_lookup_map(
    {
        SponsorType.UNIVERSITY: [
            "univ(?:ersity)?",
            "univ(?:ersities)?",
            "college",
            "research hospital",
        ],
        SponsorType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        SponsorType.INDUSTRY: [
            "(?:bio)?pharma(?:ceutical)s?",
            "biotech(?:nology)",
            "llc",
            "corp",
        ],
        SponsorType.GOVERNMENTAL: [
            "government",
            "govt",
            "federal",
            "state",
            "us health",
            "veterans affairs",
        ],
        SponsorType.HEALTH_SYSTEM: [
            "hospital",
            "health system",
            "healthcare",
            "medical system",
        ],
    }
)


class TrialStatus(IntEnum):
    PRE_ENROLLMENT = "PRE_ENROLLMENT"  # Active, not recruiting, Not yet recruiting
    ENROLLMENT = "ENROLLMENT"  # Recruiting, Enrolling by invitation
    WITHDRAWN = "WITHDRAWN"  # Withdrawn
    SUSPENDED = "SUSPENDED"  # Suspended
    TERMINATED = "TERMINATED"  # Terminated
    COMPLETED = "COMPLETED"  # Completed
    UNKNOWN = "UNKNOWN"
    NA = "N/A"

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


class TrialPhase(IntEnum):
    EARLY_PHASE_1 = "EARLY_PHASE_1"  # Early Phase 1
    PHASE_1 = "PHASE_1"  # Phase 1
    PHASE_1_2 = "PLASE_1_2"  # Phase 1/Phase 2
    PHASE_2 = "PHASE_2"  # Phase 2
    PHASE_2_3 = "PHASE_2_3"  # Phase 2/Phase 3
    PHASE_3 = "PHASE_3"  # Phase 3
    PHASE_4 = "PHASE_4"  # Phase 4
    NA = "N/A"  # Not Applicable
    UNKNOWN = "UNKNOWN"  # Unknown status

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


class TerminationReason(IntEnum):
    FUTILITY = "FUTILITY"
    SAFETY = "SAFETY"
    BUSINESS = "BUSINESS"
    ENROLLMENT = "ENROLLMENT"
    INVESTIGATOR = "INVESTIGATOR"
    FUNDING = "FUNDING"
    COVID = "COVID"
    OVERSIGHT = "OVERSIGHT"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    PROTOCOL_REVISION = "PROTOCOL_REVISION"
    FEASIBILITY = "FEASIBILITY"
    NOT_SAFETY = "NOT_SAFETY"
    LOGISTICS = "LOGISTICS"
    NOT_EFFICACY = "NOT_EFFICACY"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value):
        reason = classify_string(
            value,  # type: ignore
            TERMINATION_KEYWORD_MAP,
            TerminationReason.OTHER,
        )
        return reason[0]


TERMINATION_KEYWORD_MAP = create_lookup_map(
    {
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
            "lost to follow up",  # TODO: is this really a safety issue?
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
        TerminationReason.ENROLLMENT: [
            "accruals?",
            "enroll?(?:ment|ed)?",
            "inclusions?",
            "recruit(?:ment|ing)?s?",
            "lack of (?:eligibile )?(?:participants?|subjects?|patients?)",
        ],
        TerminationReason.INVESTIGATOR: ["investigator", "PI"],
        TerminationReason.FUNDING: ["funding", "resources", "budget", "financial"],
        TerminationReason.COVID: ["covid", "coronavirus", "pandemic"],
        TerminationReason.OVERSIGHT: [
            "IRB",
            "ethics",  # "ethics committee",
            "Institutional Review Board",
            "certification",
            "FDA",
        ],
        TerminationReason.SUPPLY_CHAIN: [
            "supply",
            "unavalaible",
            "shortage",
            "manufacturing",
        ],
        TerminationReason.PROTOCOL_REVISION: [
            "revision",
            "change in (?:study )?protocol",
        ],
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
)


class BaseTrial(TypedDict):
    """
    Base trial info

    TODO:
    - conditions -> disesases ?
    - interventions -> compounds ?
    """

    nct_id: str
    conditions: list[str]
    end_date: date
    enrollment: int
    interventions: list[str]
    last_updated_date: date
    sponsor: str
    start_date: date
    title: str


class TrialRecord(BaseTrial):
    # TODO non-inferiority, superiority
    # TODO comparator type (ACTIVE CONTROL, PLACEBO, etc.)
    # TODO dropout
    # TODO primary endpoint
    design: str
    masking: str
    phase: str
    purpose: str
    randomization: str
    status: str


class TrialSummary(BaseTrial):
    """
    Patent trial info
    """

    design: TrialDesign
    duration: int  # in days
    masking: TrialMasking
    phase: TrialPhase
    purpose: TrialPurpose
    randomization: TrialRandomization
    sponsor_type: SponsorType
    status: TrialStatus
    termination_reason: TerminationReason


def is_trial_record(trial: dict) -> TypeGuard[TrialSummary]:
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


def is_trial_summary(trial: dict) -> TypeGuard[TrialSummary]:
    """
    Check if dict is a trial record
    """
    _is_trial_record = is_trial_record(trial)
    return (
        _is_trial_record
        and "duration" in trial
        and "design" in trial
        and "masking" in trial
        and "purpose" in trial
        and "randomization" in trial
        and "termination_reason" in trial
        and "sponsor_type" in trial
    )


def is_trial_summary_list(trials: list[dict]) -> TypeGuard[list[TrialSummary]]:
    """
    Check if list of trial records
    """
    return all(is_trial_record(trial) for trial in trials)


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
    - etc

    TODO:
    - primary endpoint
    """
    if not is_trial_record(trial):
        raise ValueError("Invalid trial record")

    return cast(
        TrialSummary,
        {
            **trial,
            "conditions": dedup(trial["conditions"]),
            "design": TrialDesign(trial["design"]),
            "duration": __calc_duration(trial["start_date"], trial["end_date"]),
            "masking": TrialMasking(trial["masking"]),
            "phase": TrialPhase(trial["phase"]),
            "purpose": TrialPurpose(trial["purpose"]),
            "randomization": TrialRandomization(trial["randomization"]),
            "sponsor_type": SponsorType(trial["sponsor"]),  # type: ignore
            "status": TrialStatus(trial["status"]),
        },
    )
