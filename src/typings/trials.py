from datetime import date, datetime
from typing import Sequence, TypeGuard, TypedDict, cast
import logging

from constants.company import COMPANY_STRINGS, LARGE_PHARMA_KEYWORDS
from core.ner.classifier import classify_string, create_lookup_map
from utils.classes import ByDefinitionOrderEnum

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrialRandomization(ByDefinitionOrderEnum):
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


class TrialMasking(ByDefinitionOrderEnum):
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


class TrialPurpose(ByDefinitionOrderEnum):
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


class TrialDesign(ByDefinitionOrderEnum):
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


class SponsorType(ByDefinitionOrderEnum):
    INDUSTRY_LARGE = "INDUSTRY_LARGE"  # big pharma
    INDUSTRY = "INDUSTRY"
    UNIVERSITY = "UNIVERSITY"
    GOVERNMENTAL = "GOVERNMENTAL"
    HEALTH_SYSTEM = "HEALTH_SYSTEM"
    FOUNDATION = "FOUNDATION"
    OTHER_ORGANIZATION = "OTHER_ORGANIZATION"
    INDIVIDUAL = "INDIVIDUAL"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value):
        reason = classify_string(value, SPONSOR_KEYWORD_MAP, cls.OTHER)  # type: ignore
        res = reason[0]
        return res


SPONSOR_KEYWORD_MAP = create_lookup_map(
    {
        SponsorType.UNIVERSITY: [
            "univ(?:ersit(?:y|ies))?",
            "colleges?",
            "research hospitals?",
            "institute?s?",
            "schools?",
            "NYU",
            "Universitaire?s?",
            # "l'Université",
            # "Université",
            "Universita(?:ri)?",
            "education",
            "Universidad",
        ],
        SponsorType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        SponsorType.INDUSTRY: [
            *COMPANY_STRINGS,
            "laboratories",
            "Procter and Gamble",
            "3M",
            "Neuroscience$",
            "associates",
            "medical$",
        ],
        SponsorType.GOVERNMENTAL: [
            "government",
            "govt",
            "federal",
            "national",
            "state",
            "us health",
            "veterans affairs",
            "NIH",
            "VA",
            "European Organisation",
            "EORTC",
            "Assistance Publique",
            "FDA",
            "Bureau",
            "Authority",
        ],
        SponsorType.HEALTH_SYSTEM: [
            "healthcare",
            "(?:medical|cancer|health) (?:center|centre|system|hospital)s?",
            "clinics?",
            "districts?",
        ],
        SponsorType.FOUNDATION: ["foundatations?", "trusts?"],
        SponsorType.OTHER_ORGANIZATION: [
            "Research Network",
            "Alliance",
            "Group$",
            "research cent(?:er|re)s?",
        ],
        SponsorType.INDIVIDUAL: [r"M\.?D\.?"],
    }
)


class TrialStatus(ByDefinitionOrderEnum):
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


class TrialPhase(ByDefinitionOrderEnum):
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


class TerminationReason(ByDefinitionOrderEnum):
    FUTILITY = "FUTILITY"
    SAFETY = "SAFETY"
    NOT_SAFETY = "NOT_SAFETY"
    NOT_FUTILITY = "NOT_FUTILITY"
    BUSINESS = "BUSINESS"
    SUPPLY_CHAIN = "SUPPLY_CHAIN"
    LOGISTICS = "LOGISTICS"
    ENROLLMENT = "ENROLLMENT"
    FEASIBILITY = "FEASIBILITY"
    INVESTIGATOR = "INVESTIGATOR"
    FUNDING = "FUNDING"
    COVID = "COVID"
    OVERSIGHT = "OVERSIGHT"
    PROTOCOL_REVISION = "PROTOCOL_REVISION"
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
        TerminationReason.NOT_FUTILITY: [
            "not related to efficacy",
            "No Safety or Efficacy Concerns",
        ],
        TerminationReason.LOGISTICS: ["logistics", "logistical"],
    }
)


class HypothesisType(ByDefinitionOrderEnum):
    SUPERIORITY = "SUPERIORITY"
    NON_INFERIORITY = "NON_INFERIORITY"
    EQUIVALENCE = "EQUIVALENCE"
    MULTIPLE = "MULTIPLE"
    NON_SUPERIORITY = "NON_SUPERIORITY"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.UNKNOWN
        if isinstance(value, Sequence):
            if len(value) == 0:
                return cls.UNKNOWN
            if len(value) == 1:
                return HYPOTHESIS_TYPE_KEYWORD_MAP[value[0]]
            else:
                return cls.MULTIPLE
        else:
            logger.warning("Hypothesis type is not a sequence: %s", value)
            return cls.OTHER


HYPOTHESIS_TYPE_KEYWORD_MAP = {
    "Superiority": HypothesisType.SUPERIORITY,
    "Superiority or Other": HypothesisType.SUPERIORITY,
    "Superiority or Other (legacy)": HypothesisType.SUPERIORITY,
    "Non-Inferiority": HypothesisType.NON_INFERIORITY,
    "Equivalence": HypothesisType.EQUIVALENCE,
    "Non-Inferiority or Equivalence": HypothesisType.NON_SUPERIORITY,
    "Non-Inferiority or Equivalence (legacy)": HypothesisType.NON_SUPERIORITY,
    "Other": HypothesisType.OTHER,
}


class ComparisonType(ByDefinitionOrderEnum):
    ACTIVE = "ACTIVE"
    PLACEBO = "PLACEBO"
    NO_INTERVENTION = "NO_INTERVENTION"
    UNKNOWN = "UNKNOWN"
    NA = "NA"
    OTHER = "OTHER"

    @classmethod
    def find(cls, value, design: TrialDesign | None = None):
        if isinstance(value, Sequence):
            if len(value) == 0:
                return cls.UNKNOWN
            if "Active Comparator" in value:
                if "Experimental" in value:
                    return cls.ACTIVE
                if (
                    len(value) == 1 or "Other" in value
                ) and design != TrialDesign.SINGLE_GROUP:
                    return cls.PLACEBO  # just guessing
            if "Placebo Comparator" in value or "Sham Comparator" in value:
                return cls.PLACEBO
            if "No Intervention" in value:
                return cls.NO_INTERVENTION
            if "Experimental" in value:
                if design == TrialDesign.SINGLE_GROUP:
                    return cls.NA
                if design in [
                    TrialDesign.PARALLEL,
                    TrialDesign.CROSSOVER,
                    TrialDesign.FACTORIAL,
                ]:
                    # assume only experimental given parallel/xover or factorial, assume that means active comp
                    return cls.ACTIVE
                return cls.UNKNOWN
            if "Other" in value:
                if design == TrialDesign.SINGLE_GROUP:
                    return cls.NA
                return cls.OTHER
            if "No Intervention" in value:
                return cls.NA
            return cls.UNKNOWN
        else:
            logger.warning("Comparison type is not a sequence: %s", value)
            return cls.UNKNOWN


class BaseTrial(TypedDict):
    """
    Base trial info
    """

    nct_id: str
    arm_types: list[str]
    conditions: list[str]
    dropout_count: int
    dropout_reasons: list[str]
    end_date: date
    enrollment: int
    hypothesis_types: list[str]
    interventions: list[str]
    last_updated_date: date
    mesh_conditions: list[str]
    primary_outcomes: list[str]
    sponsor: str
    start_date: date
    title: str


class TrialRecord(BaseTrial):
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

    comparison_type: ComparisonType
    design: TrialDesign
    duration: int  # in days
    hypothesis_type: HypothesisType
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
        and "arm_types" in trial
        and "end_date" in trial
        and "start_date" in trial
        and "last_updated_date" in trial
        and "status" in trial
        and "title" in trial
        and "phase" in trial
        and "conditions" in trial
        and "mesh_conditions" in trial
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
        and "comparison_type" in trial
        and "duration" in trial
        and "design" in trial
        and "hypothesis_type" in trial
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


def dict_to_trial_summary(trial: dict) -> TrialSummary:
    """
    Get trial summary from db record

    - formats start and end date
    - calculates duration
    - etc
    """
    if not is_trial_record(trial):
        raise ValueError("Invalid trial record")

    design = TrialDesign(trial["design"])
    return cast(
        TrialSummary,
        {
            **trial,
            "comparison_type": ComparisonType.find(trial["arm_types"], design),
            "design": design,
            "duration": __calc_duration(trial["start_date"], trial["end_date"]),
            "hypothesis_type": HypothesisType(trial["hypothesis_types"]),
            "masking": TrialMasking(trial["masking"]),
            "phase": TrialPhase(trial["phase"]),
            "purpose": TrialPurpose(trial["purpose"]),
            "randomization": TrialRandomization(trial["randomization"]),
            "sponsor_type": SponsorType(trial["sponsor"]),
            "status": TrialStatus(trial["status"]),
            "termination_reason": TerminationReason(trial["termination_reason"]),
        },
    )
