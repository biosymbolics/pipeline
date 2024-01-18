from dataclasses import dataclass
from datetime import datetime
from typing import Sequence
import logging
import re
from prisma.enums import (
    BiomedicalEntityType,
    ComparisonType,
    HypothesisType,
    OwnerType,
    TerminationReason,
    TrialDesign,
    TrialMasking,
    TrialPhase,
    TrialPurpose,
    TrialRandomization,
    TrialStatus,
)

from core.ner.classifier import classify_string
from data.domain.biomedical.trials import TERMINATION_KEYWORD_MAP
from typings.core import Dataclass
from typings.documents.trials import TrialStatusGroup
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

    id: str
    acronym: str | None
    arm_count: int
    arm_types: list[str]
    dropout_count: int
    dropout_reasons: list[str]
    end_date: datetime
    enrollment: int
    hypothesis_types: list[str]
    interventions: list[str]
    intervention_types: list[str]
    last_updated_date: datetime
    outcomes: list[str]
    sponsor: str
    start_date: datetime
    time_frames: list[str]
    title: str
    termination_description: str | None


@dataclass(frozen=True)
class TrialRecord(BaseTrial):
    design: str
    indications: list[str]
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


class TrialDesignParser:
    @staticmethod
    def find_from_record(record: TrialRecord) -> TrialDesign:
        """
        Find trial design from record
        """

        if TrialPhaseParser.is_phase_1(TrialPhaseParser.find(record.phase)):
            if has_intersection(re.split("[ -]", record.title.lower()), DOSE_TERMS):
                return TrialDesign.DOSING

        provisional_design = TrialDesignParser.find(record.design)

        if provisional_design == TrialDesign.SINGLE_GROUP:
            # if more than one arm or has control, it's not single group.
            # same if trial is blinded or randomized (these fields tend to be more accurate than design)
            # we'll just assume it is parallel, but it could be crossover, dosing, etc.
            randomization = TrialRandomizationParser.find(record.randomization)
            comparison = ComparisonTypeParser.find_from_record(record)
            blinding = TrialBlinding.find(TrialMaskingParser.find(record.masking))
            if (
                not comparison in [ComparisonType.UNKNOWN, ComparisonType.NO_CONTROL]
                or (record.arm_count or 0) > 1
                or blinding == TrialBlinding.BLINDED
                or randomization == TrialRandomization.RANDOMIZED
            ):
                return TrialDesign.PARALLEL
            return TrialDesign.SINGLE_GROUP

        return provisional_design

    @staticmethod
    def find(value: str) -> TrialDesign:
        design_term_map = {
            "Parallel Assignment": TrialDesign.PARALLEL,
            "Crossover Assignment": TrialDesign.CROSSOVER,
            "Factorial Assignment": TrialDesign.FACTORIAL,
            "Sequential Assignment": TrialDesign.SEQUENTIAL,
            "Single Group Assignment": TrialDesign.SINGLE_GROUP,
            "": TrialDesign.NA,
        }
        if value in design_term_map:
            return design_term_map[value]
        return TrialDesign.UNKNOWN


class TrialRandomizationParser:
    @staticmethod
    def find(value: str, design: TrialDesign | None = None):
        rand_term_map = {
            "Randomized": TrialRandomization.RANDOMIZED,
            "Non-Randomized": TrialRandomization.NON_RANDOMIZED,
            "N/A": TrialRandomization.NA,
        }
        if design == TrialDesign.SINGLE_GROUP:
            # if single group, let's call it randomization=NA
            return TrialRandomization.NA
        if value in rand_term_map:
            return rand_term_map[value]
        return TrialRandomization.UNKNOWN


class InterventionTypeParser:
    @staticmethod
    def find(values: Sequence[str]) -> BiomedicalEntityType:
        intervention_type_term_map = {
            "Behavioral": BiomedicalEntityType.BEHAVIORAL,
            "Biological": BiomedicalEntityType.PHARMACOLOGICAL,
            "Combination Product": BiomedicalEntityType.COMBINATION,
            "Device": BiomedicalEntityType.DEVICE,
            "Diagnostic Test": BiomedicalEntityType.DIAGNOSTIC,
            "Dietary Supplement": BiomedicalEntityType.DIETARY,
            "Drug": BiomedicalEntityType.PHARMACOLOGICAL,
            "Genetic": BiomedicalEntityType.PHARMACOLOGICAL,
            "Other": BiomedicalEntityType.OTHER,
            "Procedure": BiomedicalEntityType.PROCEDURE,
            "Radiation": BiomedicalEntityType.PROCEDURE,
        }
        intervention_types = [intervention_type_term_map[v] for v in values]
        if BiomedicalEntityType.PHARMACOLOGICAL in intervention_types:
            return BiomedicalEntityType.PHARMACOLOGICAL
        if BiomedicalEntityType.DEVICE in intervention_types:
            return BiomedicalEntityType.DEVICE
        if BiomedicalEntityType.BEHAVIORAL in intervention_types:
            return BiomedicalEntityType.BEHAVIORAL
        if BiomedicalEntityType.DIAGNOSTIC in intervention_types:
            return BiomedicalEntityType.DIAGNOSTIC
        if BiomedicalEntityType.PROCEDURE in intervention_types:
            return BiomedicalEntityType.PROCEDURE
        if BiomedicalEntityType.DIETARY in intervention_types:
            return BiomedicalEntityType.DIETARY

        if len(intervention_types) == 0:
            return BiomedicalEntityType.UNKNOWN
        return intervention_types[0]


class TrialMaskingParser:
    @staticmethod
    def is_blinded(tm: TrialMasking) -> bool:
        return tm not in [TrialMasking.NONE, TrialMasking.UNKNOWN]

    @staticmethod
    def find(value: str) -> TrialMasking:
        masking_term_map = {
            "None (Open Label)": TrialMasking.NONE,
            "Single": TrialMasking.SINGLE,
            "Double": TrialMasking.DOUBLE,
            "Triple": TrialMasking.TRIPLE,
            "Quadruple": TrialMasking.QUADRUPLE,
        }
        if value in masking_term_map:
            return masking_term_map[value]
        return TrialMasking.UNKNOWN


class TrialBlinding(ByDefinitionOrderEnum):
    BLINDED = "BLINDED"
    UNBLINDED = "UNBLINDED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find(cls, value: TrialMasking):
        if TrialMaskingParser.is_blinded(value):
            return cls.BLINDED
        return cls.UNBLINDED


class TrialPurposeParser:
    @staticmethod
    def find(value: str) -> TrialPurpose:
        purpose_term_map = {
            "Treatment": TrialPurpose.TREATMENT,
            "Prevention": TrialPurpose.PREVENTION,
            "Diagnostic": TrialPurpose.DIAGNOSTIC,
            "Basic Science": TrialPurpose.BASIC_SCIENCE,
            "Supportive Care": TrialPurpose.SUPPORTIVE_CARE,
            "Device Feasibility": TrialPurpose.DEVICE,
            "Educational/Counseling/Training": TrialPurpose.OTHER,
            "Health Services Research": TrialPurpose.OTHER,
            "Screening": TrialPurpose.OTHER,
        }
        if value in purpose_term_map:
            return purpose_term_map[value]
        return TrialPurpose.UNKNOWN


class TrialStatusParser:
    @staticmethod
    def find(value: str):
        """
        Example:
        TrialStatus("Active, not recruiting") -> TrialStatus.PRE_ENROLLMENT
        """
        status_term_map = {
            "Active, not recruiting": TrialStatus.PRE_ENROLLMENT,
            "Not yet recruiting": TrialStatus.PRE_ENROLLMENT,
            "Recruiting": TrialStatus.ENROLLMENT,
            "Enrolling by invitation": TrialStatus.ENROLLMENT,
            "Withdrawn": TrialStatus.WITHDRAWN,
            "Suspended": TrialStatus.SUSPENDED,
            "Terminated": TrialStatus.TERMINATED,
            "Completed": TrialStatus.COMPLETED,
            "Not Applicable": TrialStatus.NA,
        }
        if value in status_term_map:
            return status_term_map[value]
        return TrialStatus.UNKNOWN

    @staticmethod
    def get_parent(ts: TrialStatus) -> TrialStatusGroup | str:
        if ts in (TrialStatus.PRE_ENROLLMENT, TrialStatus.ENROLLMENT):
            return TrialStatusGroup.ONGOING
        if ts in (TrialStatus.WITHDRAWN, TrialStatus.SUSPENDED, TrialStatus.TERMINATED):
            return TrialStatusGroup.STOPPED
        if ts == TrialStatus.COMPLETED:
            return TrialStatusGroup.COMPLETED
        return TrialStatusGroup.UNKNOWN


class TrialPhaseParser:
    @staticmethod
    def is_phase_1(tp: TrialPhase) -> bool:
        return tp in [
            TrialPhase.EARLY_PHASE_1,
            TrialPhase.PHASE_1,
            TrialPhase.PHASE_1_2,
        ]

    @staticmethod
    def find(value: str) -> TrialPhase:
        """
        Example:
        TrialStatus("Active, not recruiting") -> TrialStatus.PRE_ENROLLMENT
        """
        phase_term_map = {
            "Early Phase 1": TrialPhase.EARLY_PHASE_1,
            "Phase 1": TrialPhase.PHASE_1,
            "Phase 1/Phase 2": TrialPhase.PHASE_1_2,
            "Phase 2": TrialPhase.PHASE_2,
            "Phase 2/Phase 3": TrialPhase.PHASE_2_3,
            "Phase 3": TrialPhase.PHASE_3,
            "Phase 4": TrialPhase.PHASE_4,
            "Not Applicable": TrialPhase.NA,
            "Approved": TrialPhase.APPROVED,
        }
        if value in phase_term_map:
            return phase_term_map[value]
        return TrialPhase.UNKNOWN


class TerminationReasonParser:
    @staticmethod
    def find(value: str | None):
        if value is None:
            return TerminationReason.NA
        reason = classify_string(
            value,
            TERMINATION_KEYWORD_MAP,
            TerminationReason.OTHER,
        )
        return reason[0]


class HypothesisTypeParser:
    @staticmethod
    def find(value) -> HypothesisType:
        if value is None:
            return HypothesisType.UNKNOWN
        if isinstance(value, Sequence):
            if len(value) == 0:
                return HypothesisType.UNKNOWN
            if len(value) == 1:
                return HYPOTHESIS_TYPE_KEYWORD_MAP[value[0]]
            else:
                return HypothesisType.MULTIPLE
        else:
            logger.warning("Hypothesis type is not a sequence: %s", value)
            return HypothesisType.OTHER


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


class ComparisonTypeParser:
    @staticmethod
    def is_intervention_match(
        intervention_names: list[str],
        comp_type: str | ComparisonType,
    ) -> bool:
        intervention_re: dict[str | ComparisonType, str] = {
            ComparisonType.DOSE: get_or_re(DOSE_TERMS),
            ComparisonType.ACTIVE: r"(?:active comparator|standard|routine|conventional|\bSOC\b)",
            ComparisonType.PLACEBO: r"(?:placebo|sham|comparator|control\b)",
        }
        pattern = intervention_re.get(comp_type) or False

        def is_match(pattern, it):
            return re.search(pattern, it, re.IGNORECASE) is not None

        return any([is_match(pattern, it) for it in intervention_names])

    @staticmethod
    def find_from_interventions(
        intervention_names: list[str], default: ComparisonType
    ) -> ComparisonType:
        """
        As a last resort, look at the intervention names to determine the comparison type
        """
        if all(
            [
                ComparisonTypeParser.is_intervention_match(
                    intervention_names, ComparisonType.DOSE
                )
            ]
        ):
            return ComparisonType.DOSE
        if ComparisonTypeParser.is_intervention_match(
            intervention_names, ComparisonType.ACTIVE
        ):
            return ComparisonType.ACTIVE
        if ComparisonTypeParser.is_intervention_match(
            intervention_names, ComparisonType.PLACEBO
        ):
            return ComparisonType.PLACEBO
        return default

    @staticmethod
    def find_from_record(record: TrialRecord) -> ComparisonType:
        """
        Find comparison type from record
        NOTE: does not use trial design (avoid recursive loop)
        """
        return ComparisonTypeParser.find(record.arm_types, record.interventions)

    @staticmethod
    def find(
        arm_types: list[str],
        interventions: list[str],
        design: TrialDesign | None = None,
    ) -> ComparisonType:
        if not isinstance(arm_types, Sequence):
            logger.warning("Comparison type is not a sequence: %s", arm_types)
            return ComparisonType.UNKNOWN

        if design == TrialDesign.DOSING:
            return ComparisonType.DOSE
        if design == TrialDesign.SINGLE_GROUP:
            return ComparisonType.NO_CONTROL
        if len(arm_types) == 0:
            return ComparisonTypeParser.find_from_interventions(
                interventions, ComparisonType.UNKNOWN
            )
        if has_intersection(["Placebo Comparator", "Sham Comparator"], arm_types):
            return ComparisonType.PLACEBO
        if "Active Comparator" in arm_types:
            # after placebo, because sometimes "experimental" is meant by "active"
            return ComparisonType.ACTIVE
        if "No Intervention" in arm_types:
            return ComparisonType.NO_INTERVENTION
        if "Experimental" in arm_types:
            # all experimental
            if len(arm_types) == 1:
                return ComparisonTypeParser.find_from_interventions(
                    interventions, ComparisonType.UNKNOWN
                )
        if "Other" in arm_types:
            return ComparisonTypeParser.find_from_interventions(
                interventions, ComparisonType.OTHER
            )
        return ComparisonType.UNKNOWN


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
    intervention_type: BiomedicalEntityType
    masking: TrialMasking
    max_timeframe: int | None  # in days
    phase: TrialPhase
    purpose: TrialPurpose
    randomization: TrialRandomization
    sponsor_type: OwnerType
    termination_reason: TerminationReason
    status: TrialStatus
    url: str


def calc_duration(start_date: datetime | None, end_date: datetime | None) -> int:
    """
    Calculate duration in days
    """
    if not start_date:
        return -1
    if not end_date:
        return (datetime.today().date() - start_date.date()).days

    return (end_date - start_date).days
