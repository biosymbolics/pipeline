"""
Patent types
"""

from dataclasses import dataclass
from datetime import date
from typing import Any
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
    def find(
        cls, assignees: list[str], is_stale: bool, financials_map: dict[str, Company]
    ) -> tuple["AvailabilityLikelihood", str]:
        """
        Find availability likelihood
        """
        if len(financials_map) == 0:
            return (AvailabilityLikelihood.UNKNOWN, "No financials provided.")  # type: ignore

        # mark all patents of troubled companies as "possibly" available
        is_troubled = any(
            company in financials_map and financials_map[company].is_troubled
            for company in assignees
        )

        explanation = f"""
            Patent is owned by a financially troubled company: {is_troubled}.
            Patent is stale: {is_stale}.
        """

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
