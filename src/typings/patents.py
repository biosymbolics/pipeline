"""
Patent types
"""

from dataclasses import dataclass
from datetime import date

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


@dataclass(frozen=True)
class ScoredPatentApplication(PatentApplication):
    adj_patent_years: int
    availability_score: int
    search_rank: float
    probability_of_success: float
    suitability_score: float
    suitability_score_explanation: str


SuitabilityScoreMap = dict[str, float]
