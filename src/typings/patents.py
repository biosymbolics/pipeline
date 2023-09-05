"""
Patent types
"""

from datetime import date
from typing import TypedDict


class PatentBasicInfo(TypedDict):
    """
    Patent basic info object as per Google Patents API / local modifications
    """

    abstract: str
    application_number: str
    assignees: list[str]
    attributes: list[str]  # keyof typeof PATENT_ATTRIBUTE_MAP # ATTRIBUTE_FIELD
    family_id: str
    ipc_codes: list[str]
    patent_years: int
    priority_date: date
    # publication_date: date
    publication_number: str
    score: float
    title: str
    url: str


class PatentApplication(PatentBasicInfo):
    """
    Patent application object as per Google Patents API / local modifications
    """

    compounds: list[str]
    country: str
    diseases: list[str]
    genes: list[str]  # remove?
    embeddings: list[float]
    # grant_date: date
    # filing_date: date
    inventors: list[str]
    last_trial_status: str
    last_trial_update: date
    max_trial_phase: str
    mechanisms: list[str]
    nct_ids: list[str]
    similar_patents: list[str]
    top_terms: list[str]  # from GPR table


class ApprovedPatentApplication(PatentApplication):
    is_approved: str
    brand_name: str
    generic_name: str
    approval_date: date
    indication: str


SuitabilityScoreMap = dict[str, float]
