"""
Patent client
"""
from typing import Sequence
import logging

from clients import select_from_bg

from .constants import COMPOSITION_OF_MATTER_IPC_CODES, METHOD_OF_USE_IPC_CODES
from .formatting import format_search_result
from .utils import get_max_priority_date
from .types import PatentBasicInfo

SEARCH_RETURN_FIELDS = [
    "applications.publication_number",
    "priority_date",
    "title",
    "abstract",
    # "application_kind",
    "application_number",
    "assignees",
    # "cited_by",
    "country",
    "family_id",
    # "cpc_codes",
    # "embedding_v1 as embeddings",
    # "filing_date",
    # "grant_date",
    "inventors",
    "ipc_codes",
    # "publication_date",
    # "similar",
    "top_terms",
    "url",
]

# composition of matter filter
COM_FILTER = (
    "("
    "SELECT COUNT(1) FROM UNNEST(ipc_codes) AS ipc "
    f"JOIN UNNEST({COMPOSITION_OF_MATTER_IPC_CODES}) AS com_code "  # composition of matter
    "ON starts_with(ipc, com_code)"
    ") > 0"
)


def search(terms: Sequence[str]) -> Sequence[PatentBasicInfo]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter

    Args:
        terms (list[str]): list of terms to search for

    Returns: a list of matching patent applications

    TODO: doesn't actually match typing - in particular, need to aggregate annotations.

    Example:
        >>> search(['asthma', 'astrocytoma'])
    """
    lower_terms = [term.lower() for term in terms]
    max_priority_date = get_max_priority_date(10)
    fields = ",".join(SEARCH_RETURN_FIELDS)
    query = (
        "WITH filtered_entities AS ( "
        "SELECT * FROM patents.annotations a, UNNEST(a.annotations) as annotation "
        f"WHERE annotation.term IN UNNEST({lower_terms}) "
        ") "
        f"SELECT {fields} "
        "FROM patents.applications AS applications "
        "JOIN filtered_entities AS entities "
        "ON applications.publication_number = entities.publication_number "
        "WHERE "
        f"priority_date > {max_priority_date} "  # min patent life
        f"AND {COM_FILTER} "  # composition of matter, no method of use
        "ORDER BY priority_date DESC "
        "limit 2000"
    )
    results = select_from_bg(query)
    return format_search_result(results)


def autocomplete_terms(string: str) -> list[str]:
    """
    Fetch all entities from patents.entity_list
    Used to update the entity list in the app

    Args:
        string (str): string to search for

    Returns: a list of matching terms
    """
    query = f"SELECT term from patents.entity_list where term like '%{string}%' order by term asc"
    results = select_from_bg(query)
    return [result["term"] for result in results]
