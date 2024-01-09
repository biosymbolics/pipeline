"""
Patent types
"""

from dataclasses import dataclass
from typing import Any, Sequence, TypedDict
from pydash import compact
from prisma.models import Patent

from typings.companies import CompanyFinancials
from typings.core import Dataclass, inject_fields
from utils.classes import ByDefinitionOrderEnum


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
        financial_map: dict[str, CompanyFinancials],
    ) -> list[str]:
        """
        Compose explanation for availability likelihood
        """
        explanations = compact(
            [
                f"{company} has some financial signal ({financial_map[company]})."
                for company in troubled_assignees
            ]
        )

        return explanations

    @classmethod
    def find_from_record(
        cls,
        record: dict[str, Any],
        financial_map: dict[str, CompanyFinancials],
    ) -> tuple["AvailabilityLikelihood", str]:
        """
        Find availability likelihood from record
        """

        names = [o["name"] for o in record.get("assignees") or []]

        return cls.find(
            names,
            financial_map,
        )

    @classmethod
    def find(
        cls,
        assignee_names: list[str],
        financial_map: dict[str, CompanyFinancials],
    ) -> tuple["AvailabilityLikelihood", str]:
        """
        Find availability likelihood
        """
        if len(financial_map) == 0:
            return (AvailabilityLikelihood.UNKNOWN, "No financials provided.")  # type: ignore

        # mark all patents of troubled companies as "possibly" available
        troubled_assignees = [
            owner
            for owner in assignee_names
            if owner in financial_map and financial_map[owner].is_troubled
        ]
        is_troubled = len(troubled_assignees) > 0

        troubled_detail = cls.compose_financial_explanation(
            troubled_assignees, financial_map
        )

        if is_troubled:
            explanation = "\n".join(troubled_detail)
            return (AvailabilityLikelihood.POSSIBLE, explanation)  # type: ignore

        return (AvailabilityLikelihood.UNKNOWN, "N/A")  # type: ignore


@dataclass
@inject_fields(Patent, Dataclass)
class ScoredPatent:
    adj_patent_years: int
    availability_likelihood: AvailabilityLikelihood
    availability_explanation: str
    availability_score: float
    exemplar_similarity: float
    patent_years: int
    probability_of_success: float
    reformulation_score: float
    score: float
    search_rank: float
    suitability_score: float
    suitability_score_explanation: str


SuitabilityScoreMap = dict[str, float]
