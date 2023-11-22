"""
Patent types
"""

from dataclasses import dataclass
from datetime import date
from typing import Any, Sequence

from pydash import compact
from typings.companies import Company

from utils.classes import ByDefinitionOrderEnum

from .core import Dataclass


@dataclass(frozen=True)
class PatentBasicInfo(Dataclass):
    """
    Patent basic info object as per Google Patents API / local modifications
    """

    abstract: str
    application_number: str
    assignees: list[str]
    attributes: list[str]  # keyof typeof get_patent_attribute_map()
    family_id: str
    ipc_codes: list[str]
    patent_years: int
    priority_date: date
    publication_number: str
    score: float
    title: str
    url: str


@dataclass(frozen=True)
class PatentApplication(PatentBasicInfo):
    """
    Patent application object as per Google Patents API / local modifications
    """

    biologics: list[str]
    compounds: list[str]
    country: str
    devices: list[str]
    diseases: list[str]
    embeddings: list[float]
    inventors: list[str]
    last_trial_status: str
    last_trial_update: date
    max_trial_phase: str
    mechanisms: list[str]
    nct_ids: list[str]
    similar_patents: list[str]

    # approved patent fields
    is_approved: str
    brand_name: str
    generic_name: str
    approval_date: date
    approval_indications: list[str]


class AvailabilityLikelihood(ByDefinitionOrderEnum):
    LIKELY = "LIKELY"
    POSSIBLE = "POSSIBLE"
    UNLIKELY = "UNLIKELY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def compose_explanation(
        cls,
        troubled_assignees: Sequence[str],
        is_stale: bool,
        financials_map: dict[str, Company],
    ) -> str:
        """
        Compose explanation for availability likelihood
        """
        explanations = compact(
            [
                *[
                    f"{company} has some financial signal ({financials_map[company]})."
                    for company in troubled_assignees
                ],
                "Patent is stale." if is_stale else None,
            ]
        )

        if len(explanations) == 0:
            return "N/A"

        return "\n".join(explanations)

    @classmethod
    def find(
        cls, assignees: list[str], is_stale: bool, financials_map: dict[str, Company]
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

        explanation = cls.compose_explanation(
            troubled_assignees, is_stale, financials_map
        )

        if is_troubled and is_stale:
            return (AvailabilityLikelihood.LIKELY, explanation)  # type: ignore

        if is_troubled or is_stale:
            return (AvailabilityLikelihood.POSSIBLE, explanation)  # type: ignore

        return (AvailabilityLikelihood.UNKNOWN, explanation)  # type: ignore


@dataclass(frozen=True)
class ScoredPatentApplication(PatentApplication):
    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)

    adj_patent_years: int
    availability_likelihood: AvailabilityLikelihood
    availability_explanation: str
    probability_of_success: float
    search_rank: float
    suitability_score: float
    suitability_score_explanation: str


SuitabilityScoreMap = dict[str, float]
