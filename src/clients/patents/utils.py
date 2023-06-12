"""
Utility functions for the patents client
"""
from typing import Optional, cast
from datetime import date

MAX_PATENT_LIFE = 20

from .types import PatentAttribute


def get_max_priority_date(min_patent_years: Optional[int] = 0) -> int:
    """
    Outputs max priority date as number YYYYMMDD (e.g. 20211031)

    Args:
        min_patent_years (Optional[int], optional): min number of years remaining on patent. Defaults to 0.
    """
    # e.g. 2021 - 20 = 2001
    priority_year = date.today().year - MAX_PATENT_LIFE

    # e.g. 2001 + min of 10 yrs remaining = 2011
    max_priority_year = priority_year + (min_patent_years or 0)

    # e.g. 2001 -> 20010000
    as_number = max_priority_year * 10000
    return as_number


def get_patent_attributes(title: str) -> list[PatentAttribute]:
    title_words = set(title.lower().split(" "))
    attributes = []
    if set(["novel"]).intersection(title_words):
        attributes.append("Novel")
    if set(["combination", "combinations", "combined"]).intersection(title_words):
        attributes.append("Combination")
    if set(["method", "methods", "system", "systems"]).intersection(title_words):
        attributes.append("Method")
    if set(
        [
            "diagnosis",
            "diagnostic",
            "diagnosing",
            "biomarker",
            "biomarkers",
            "detection",
            "marker",
            "markers",
            "monitoring",
            "risk score",
            "sensor",
            "sensors",
            "testing",
        ]
    ).intersection(title_words):
        attributes.append("Diagnostic")
    if set(
        [
            "analogue",
            "analog",
            "antibody",
            "antibodies",  # may be more biomarker
            "compound",
            "compounds",
            "compositions",
            "derivative",
            "derivatives",
            "ligand",
            "ligands",
            "substituted",
            "substitute",
            "substitutes",
            "modulator",
            "modulators",
            "inhibitor",
            "inhibitors",
            "agonist",
            "antagonist",
            "prodrug",
            "prodrugs",
        ]
    ).intersection(title_words):
        attributes.append("Compound")
    if set(["formulation", "formulations", "preparation"]).intersection(title_words):
        attributes.append("Formulation")
    return cast(list[PatentAttribute], attributes)
