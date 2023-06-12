"""
Patent client
"""
from typing import cast

from clients import execute_bg_query

from .types import PatentApplication


def search_patents(terms: list[str]) -> list[PatentApplication]:
    """
    Search patents by terms

    Args:
        terms (list[str]): list of terms to search for

    Returns: a list of matching patent applications
    """
    query = (
        "select * "
        "from "
        "patents.applications as applications, "
        "patents.entities as entities "
        "where "
        "applications.publication_number = entities.publication_number "
        f"and entities.term in UNNEST({terms}) "
    )
    results = execute_bg_query(query)
    return cast(list[PatentApplication], results)
