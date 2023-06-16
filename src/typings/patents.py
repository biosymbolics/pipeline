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
    attributes: list[str]  # keyof typeof PATENT_ATTRIBUTE_MAP
    family_id: str
    ipc_codes: list[str]
    patent_years: int
    priority_date: date
    publication_number: str
    score: float
    search_rank: float
    title: str
    url: str


class PatentApplication(PatentBasicInfo):
    """
    Patent application object as per Google Patents API / local modifications
    """

    application_kind: str
    compounds: list[str]
    country: str
    diseases: list[str]
    genes: list[str]
    effects: list[str]
    embeddings: list[float]
    grant_date: date
    filing_date: date
    indications: list[str]
    inventors: list[str]
    proteins: list[str]
    similar: list[str]
    top_terms: list[str]  # from GPR table