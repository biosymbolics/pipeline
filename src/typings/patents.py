"""
Patent types
"""

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Sequence, TypedDict
from pydash import compact
from prisma.models import Patent

from typings.companies import CompanyFinancials
from utils.classes import ByDefinitionOrderEnum

from .core import Dataclass


STALE_YEARS = 5
MAX_PATENT_LIFE = 20

PatentsTopicReport = TypedDict(
    "PatentsTopicReport",
    {"x": float, "y": float, "publication_number": str},
)


class AvailabilityLikelihood(ByDefinitionOrderEnum):
    LIKELY = "LIKELY"
    POSSIBLE = "POSSIBLE"
    UNLIKELY = "UNLIKELY"
    UNKNOWN = "UNKNOWN"

    @property
    def score(self) -> float:
        if self == AvailabilityLikelihood.LIKELY:
            return 0.25

        if self == AvailabilityLikelihood.POSSIBLE:
            return 0.15

        if self == AvailabilityLikelihood.UNLIKELY:
            return -0.5

        return 0.0

    @classmethod
    def compose_financial_explanation(
        cls,
        troubled_assignees: Sequence[str],
        financials_map: dict[str, CompanyFinancials],
    ) -> list[str]:
        """
        Compose explanation for availability likelihood
        """
        explanations = compact(
            [
                f"{company} has some financial signal ({financials_map[company]})."
                for company in troubled_assignees
            ]
        )

        return explanations

    @classmethod
    def is_patent_active(cls, record: dict[str, Any]) -> bool | None:
        if record["last_trial_update"] is None:
            return None
        return date.today() - record["last_trial_update"] < timedelta(
            days=STALE_YEARS * 365
        )

    @classmethod
    def find_from_record(
        cls,
        record: dict[str, Any],
        financials_map: dict[str, CompanyFinancials],
    ) -> tuple["AvailabilityLikelihood", str]:
        """
        Find availability likelihood from record
        """
        is_active = cls.is_patent_active(record)
        is_terminated = (
            record["termination_reason"] is not None
            and record["termination_reason"] != "NA"
        )

        return cls.find(
            record["assignees"],
            is_active,
            is_terminated,
            record["termination_reason"],
            financials_map,
        )

    @classmethod
    def find(
        cls,
        assignees: list[str],
        is_active: bool | None,
        is_terminated: bool,
        termination_reason: str | None,
        financials_map: dict[str, CompanyFinancials],
    ) -> tuple["AvailabilityLikelihood", str]:
        """
        Find availability likelihood
        """
        if len(financials_map) == 0:
            return (AvailabilityLikelihood.UNKNOWN, "No financials provided.")  # type: ignore

        # mark all patents of troubled companies as "possibly" available
        troubled_assignees = [
            company
            for company in assignees
            if company in financials_map and financials_map[company].is_troubled
        ]
        is_troubled = len(troubled_assignees) > 0

        troubled_detail = cls.compose_financial_explanation(
            troubled_assignees, financials_map
        )

        if is_terminated:
            explanation = "\n".join(
                [*troubled_detail, f"Trial terminated: {termination_reason}"]
            )
            return (AvailabilityLikelihood.POSSIBLE, explanation)  # type: ignore

        if is_active == True:
            explanation = "\n".join([*troubled_detail, f"Trial is active"])
            return (AvailabilityLikelihood.UNLIKELY, explanation)  # type: ignore

        if is_troubled or is_active == False:
            explanation = "\n".join([*troubled_detail, f"Trial is active: {is_active}"])
            return (AvailabilityLikelihood.POSSIBLE, explanation)  # type: ignore

        return (AvailabilityLikelihood.UNKNOWN, "N/A")  # type: ignore


@dataclass
class ScoredPatent(Patent):
    adj_patent_years: int
    availability_likelihood: AvailabilityLikelihood
    availability_explanation: str
    availability_score: float
    exemplar_similarity: float
    probability_of_success: float
    reformulation_score: float
    search_rank: float
    suitability_score: float
    suitability_score_explanation: str


SuitabilityScoreMap = dict[str, float]
