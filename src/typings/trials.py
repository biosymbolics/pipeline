from dataclasses import dataclass, field
from datetime import date, datetime
import random
from typing import Sequence, TypeGuard
import logging
import re

from constants.company import COMPANY_STRINGS, LARGE_PHARMA_KEYWORDS
from core.ner.classifier import classify_string, create_lookup_map
from data.domain.trials import extract_max_timeframe
from typings.core import Dataclass
from utils.classes import ByDefinitionOrderEnum
from utils.list import has_intersection
from utils.re import get_or_re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class BaseTrial(Dataclass):
    """
    Base trial info
    """

    nct_id: str
    acronym: str | None
    arm_count: int
    arm_types: list[str]
    conditions: list[str]
    dropout_count: int
    dropout_reasons: list[str]
    end_date: date
    enrollment: int
    hypothesis_types: list[str]
    intervention: str | None
    interventions: list[str]
    intervention_types: list[str]
    last_updated_date: date
    mesh_conditions: list[str]
    normalized_sponsor: str
    pharmacologic_class: str | None
    primary_outcomes: list[str]
    sponsor: str
    start_date: date
    time_frames: list[str]
    title: str
    why_stopped: str | None


@dataclass(frozen=True)
class TrialRecord(BaseTrial):
    design: str
    masking: str
    phase: str
    purpose: str
    randomization: str
    status: str


DOSE_TERMS = [
    "dose",
    "doses",
    "dosing",
    "dosage",
    "dosages",
    "mad",
    "sad",
    "pharmacokinetic",
    "pharmacokinetics",
    "pharmacodynamics",
    "titration",
    "titrating",
]


class TrialDesign(ByDefinitionOrderEnum):
    PARALLEL = "PARALLEL"
    CROSSOVER = "CROSSOVER"
    FACTORIAL = "FACTORIAL"
    SEQUENTIAL = "SEQUENTIAL"
    SINGLE_GROUP = "SINGLE_GROUP"
    DOSING = "DOSING"
    NA = "N/A"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find(cls, record: TrialRecord):
        """
        Find trial design from record
        """

        if TrialPhase(record["phase"]).is_phase_1():
            if has_intersection(re.split("[ -]", record.title.lower()), DOSE_TERMS):
                return TrialDesign.DOSING

        provisional_design = TrialDesign(record["design"])

        if provisional_design == TrialDesign.SINGLE_GROUP:
            # if more than one arm or has control, it's not single group.
            # same if trial is blinded or randomized (these fields tend to be more accurate than design)
            # we'll just assume it is parallel, but it could be crossover, dosing, etc.
            randomization = TrialRandomization.find(record.randomization)
            comparison = ComparisonType.find_from_record(record)
            blinding = TrialBlinding.find(TrialMasking(record.masking))
            if (
                not comparison in [ComparisonType.UNKNOWN, ComparisonType.NO_CONTROL]
                or (record.arm_count or 0) > 1
                or blinding == TrialBlinding.BLINDED
                or randomization == TrialRandomization.RANDOMIZED
            ):
                return TrialDesign.PARALLEL
            return TrialDesign.SINGLE_GROUP

        return provisional_design

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


class TrialRandomization(ByDefinitionOrderEnum):
    RANDOMIZED = "RANDOMIZED"
    NON_RANDOMIZED = "NON_RANDOMIZED"
    NA = "N/A"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find(cls, randomization: str, design: TrialDesign | None = None):
        if design == TrialDesign.SINGLE_GROUP:
            # if single group, let's call it randomization=NA
            return cls.NA
        return cls(randomization)

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


class InterventionType(ByDefinitionOrderEnum):
    BEHAVIORAL = "BEHAVIORAL"
    COMBINATION = "COMBINATION"
    DEVICE = "DEVICE"
    DIAGNOSTIC = "DIAGNOSTIC"
    DIETARY = "DIETARY"
    OTHER = "OTHER"
    PHARMACOLOGICAL = "PHARMACOLOGICAL"
    PROCEDURE = "PROCEDURE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def _missing_(cls, values: Sequence[str]):
        intervention_type_term_map = {
            "Behavioral": cls.BEHAVIORAL,
            "Biological": cls.PHARMACOLOGICAL,
            "Combination Product": cls.COMBINATION,
            "Device": cls.DEVICE,
            "Diagnostic Test": cls.DIAGNOSTIC,
            "Dietary Supplement": cls.DIETARY,
            "Drug": cls.PHARMACOLOGICAL,
            "Genetic": cls.PHARMACOLOGICAL,
            "Other": cls.OTHER,
            "Procedure": cls.PROCEDURE,
            "Radiation": cls.PROCEDURE,
        }
        intervention_types = [intervention_type_term_map.get(v) for v in values]
        if cls.PHARMACOLOGICAL in intervention_types:
            return cls.PHARMACOLOGICAL
        if cls.DEVICE in intervention_types:
            return cls.DEVICE
        if cls.BEHAVIORAL in intervention_types:
            return cls.BEHAVIORAL
        if cls.DIAGNOSTIC in intervention_types:
            return cls.DIAGNOSTIC
        if cls.PROCEDURE in intervention_types:
            return cls.PROCEDURE
        if cls.DIETARY in intervention_types:
            return cls.DIETARY

        return intervention_types[0]


class TrialMasking(ByDefinitionOrderEnum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    QUADRUPLE = "QUADRUPLE"
    UNKNOWN = "UNKNOWN"

    def is_blinded(self) -> bool:
        return self not in [self.NONE, self.UNKNOWN]

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


class TrialBlinding(ByDefinitionOrderEnum):
    BLINDED = "BLINDED"
    UNBLINDED = "UNBLINDED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find(cls, value: TrialMasking):
        if value.is_blinded():
            return cls.BLINDED
        return cls.UNBLINDED


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
            "Approved": cls.APPROVED,
        }
        if value in status_term_map:
            return status_term_map[value]
        return cls.UNKNOWN


class TrialPhase(ByDefinitionOrderEnum):
    EARLY_PHASE_1 = "EARLY_PHASE_1"  # Early Phase 1
    PHASE_1 = "PHASE_1"  # Phase 1
    PHASE_1_2 = "PHASE_1_2"  # Phase 1/Phase 2
    PHASE_2 = "PHASE_2"  # Phase 2
    PHASE_2_3 = "PHASE_2_3"  # Phase 2/Phase 3
    PHASE_3 = "PHASE_3"  # Phase 3
    PHASE_4 = "PHASE_4"  # Phase 4
    NA = "N/A"  # Not Applicable
    UNKNOWN = "UNKNOWN"  # Unknown status
    APPROVED = "APPROVED"

    def is_phase_1(self) -> bool:
        return self in [self.EARLY_PHASE_1, self.PHASE_1, self.PHASE_1_2]

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
            "Approved": cls.APPROVED,
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
    NA = "N/A"

    @classmethod
    def _missing_(cls, value: str | None):
        if value is None:
            return cls.NA
        reason = classify_string(
            value,
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
    NO_CONTROL = "NO_CONTROL"
    DOSE = "DOSE"
    UNKNOWN = "UNKNOWN"
    NA = "NA"
    OTHER = "OTHER"

    @classmethod
    def is_intervention_match(
        cls,
        intervention_names: list[str],
        comp_type: "str | ComparisonType",
    ) -> bool:
        intervention_re: dict[str | ComparisonType, str] = {
            cls.DOSE: get_or_re(DOSE_TERMS),
            cls.ACTIVE: r"(?:active comparator|standard|routine|conventional|\bSOC\b)",
            cls.PLACEBO: r"(?:placebo|sham|comparator|control\b)",
        }
        pattern = intervention_re.get(comp_type) or False

        def is_match(pattern, it):
            return re.search(pattern, it, re.IGNORECASE) is not None

        return any([is_match(pattern, it) for it in intervention_names])

    @classmethod
    def find_from_interventions(
        cls, intervention_names: list[str], default: "str | ComparisonType"
    ):
        """
        As a last resort, look at the intervention names to determine the comparison type
        """
        if all([cls.is_intervention_match(intervention_names, cls.DOSE)]):
            return cls.DOSE
        if cls.is_intervention_match(intervention_names, cls.ACTIVE):
            return cls.ACTIVE
        if cls.is_intervention_match(intervention_names, cls.PLACEBO):
            return cls.PLACEBO
        return default

    @classmethod
    def find_from_record(cls, record: TrialRecord):
        """
        Find comparison type from record
        NOTE: does not use trial design (avoid recursive loop)
        """
        return cls.find(record["arm_types"], record["interventions"])

    @classmethod
    def find(
        cls,
        arm_types: list[str],
        interventions: list[str],
        design: TrialDesign | None = None,
    ):
        if not isinstance(arm_types, Sequence):
            logger.warning("Comparison type is not a sequence: %s", arm_types)
            return cls.UNKNOWN

        if design == TrialDesign.DOSING:
            return cls.DOSE
        if design == TrialDesign.SINGLE_GROUP:
            return cls.NO_CONTROL
        if len(arm_types) == 0:
            return cls.find_from_interventions(interventions, cls.UNKNOWN)
        if has_intersection(["Placebo Comparator", "Sham Comparator"], arm_types):
            return cls.PLACEBO
        if "Active Comparator" in arm_types:
            # after placebo, because sometimes "experimental" is meant by "active"
            return cls.ACTIVE
        if "No Intervention" in arm_types:
            return cls.NO_INTERVENTION
        if "Experimental" in arm_types:
            # all experimental
            if len(arm_types) == 1:
                return cls.find_from_interventions(interventions, cls.UNKNOWN)
        if "Other" in arm_types:
            return cls.find_from_interventions(interventions, cls.OTHER)
        return cls.UNKNOWN


@dataclass(frozen=True)
class TrialSummary(BaseTrial):
    """
    Patent trial info
    """

    blinding: TrialBlinding
    comparison_type: ComparisonType
    design: TrialDesign
    duration: int  # in days
    hypothesis_type: HypothesisType
    intervention_type: InterventionType
    masking: TrialMasking
    max_timeframe: int | None  # in days
    phase: TrialPhase
    purpose: TrialPurpose
    randomization: TrialRandomization
    sponsor_type: SponsorType
    termination_reason: TerminationReason
    status: TrialStatus


@dataclass(frozen=True)
class ScoredTrialSummary(TrialSummary):
    score: float

    @property
    def condition(self) -> str | None:
        if len(self.mesh_conditions or []) == 0:
            return None
        return self.mesh_conditions[0]

    @property
    def instance_rollup(self) -> str | None:
        ir = self.pharmacologic_class or self.intervention
        if ir is None:
            return None
        return ir.lower()

    @property
    def dropout_percent(self) -> float:
        if not self.enrollment or not self.dropout_count:
            return 0.0
        return round(self.dropout_count / self.enrollment, 2)

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


def calc_duration(start_date: date | None, end_date: date | None) -> int:
    """
    Calculate duration in days
    """
    if not start_date:
        return -1
    if not end_date:
        return (datetime.today().date() - start_date).days

    return (end_date - start_date).days


def raw_to_trial_summary(trial: TrialRecord) -> TrialSummary:
    """
    Get trial summary from db record

    - formats start and end date
    - calculates duration
    - etc
    """
    design = TrialDesign.find(trial)
    masking = TrialMasking(trial["masking"])

    return TrialSummary(
        **{
            **trial,
            "blinding": TrialBlinding.find(masking),
            "comparison_type": ComparisonType.find(
                trial["arm_types"] or [], trial["interventions"], design
            ),
            "design": design,
            "duration": calc_duration(trial["start_date"], trial["end_date"]),  # type: ignore
            "max_timeframe": extract_max_timeframe(trial["time_frames"]),
            "hypothesis_type": HypothesisType(trial["hypothesis_types"]),
            "intervention_type": InterventionType(trial["intervention_types"]),
            "masking": masking,
            "phase": TrialPhase(trial["phase"]),
            "purpose": TrialPurpose(trial["purpose"]),
            "randomization": TrialRandomization.find(trial["randomization"], design),  # type: ignore
            "sponsor_type": SponsorType(trial["sponsor"]),
            "status": TrialStatus(trial["status"]),
            "termination_reason": TerminationReason(trial["why_stopped"]),
        },
    )


Trial = ScoredTrialSummary
