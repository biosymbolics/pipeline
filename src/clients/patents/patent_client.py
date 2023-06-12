"""
Patent client
"""
from typing import cast

from clients import select_from_bg

from .utils import get_max_priority_date
from .types import PatentApplication

SEARCH_RETURN_FIELDS = [
    "applications.publication_number",
    "abstract",
    "application_kind",
    "application_number",
    "assignees",
    # "cited_by",
    "country",
    "cpc_codes",
    # "embedding_v1 as embeddings",
    "inventors",
    "ipc_codes",
    "publication_date",
    # "similar",
    "title",
    "top_terms",
    "url",
]


def search_patents(terms: list[str]) -> list[PatentApplication]:
    """
    Search patents by terms
    Filters on priority date

    Args:
        terms (list[str]): list of terms to search for

    Returns: a list of matching patent applications
    """
    max_priority_date = get_max_priority_date(10)
    query = (
        "WITH filtered_entities AS ( "
        "SELECT * "
        "FROM patents.entities, UNNEST(annotations) as annotation "
        f"WHERE annotation.term IN UNNEST({terms}) "
        ") "
        "SELECT "
        f"{','.join(SEARCH_RETURN_FIELDS)} "
        "FROM patents.applications AS applications "
        "JOIN filtered_entities AS entities "
        "ON applications.publication_number = entities.publication_number "
        f"WHERE priority_date > {max_priority_date} "
        "limit 100"
    )
    results = select_from_bg(query)
    return cast(list[PatentApplication], results)
