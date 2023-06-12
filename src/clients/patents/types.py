"""
Patent types
"""
from datetime import date
from typing import Literal, TypedDict

PatentAttribute = Literal[
    "Combination", "Compound", "Diagnostic", "Formulation", "Method", "Novel"
]


class PatentApplication(TypedDict):
    """
    Patent application object as per Google Patents API / local modifications
    """

    abstract: str
    application_kind: str
    application_number: str
    assignees: list[str]
    compounds: list[str]
    country: str
    genes: list[str]
    effects: list[str]
    embeddings: list[float]
    grant_date: date
    family_id: str
    filing_date: date
    ipcs: list[str]
    indications: list[str]
    inventors: list[str]
    priority_date: date
    similar: list[str]
    title: str
    top_terms: list[str]  # from GPR table
