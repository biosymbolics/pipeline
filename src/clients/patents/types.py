"""
Patent types
"""
from datetime import date
from typing import Literal, TypedDict


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
    priority_date: date
    title: str
    url: str


class PatentApplication(PatentBasicInfo):
    """
    Patent application object as per Google Patents API / local modifications
    """

    application_kind: str
    compounds: list[str]
    country: str
    genes: list[str]
    effects: list[str]
    embeddings: list[float]
    grant_date: date
    filing_date: date
    indications: list[str]
    inventors: list[str]
    similar: list[str]
    top_terms: list[str]  # from GPR table
