"""
Patent client
"""
from typing import cast
import polars as pl

from clients import select_from_bg
from common.utils.date import parse_date

from .constants import COMPOSITION_OF_MATTER_IPC_CODES, METHOD_OF_USE_IPC_CODES
from .utils import get_max_priority_date
from .types import PatentApplication

SEARCH_RETURN_FIELDS = [
    "applications.publication_number",
    "title",
    "abstract",
    "priority_date",
    "application_kind",
    "application_number",
    "assignees",
    # "cited_by",
    "country",
    # "cpc_codes",
    # "embedding_v1 as embeddings",
    "filing_date",
    "grant_date",
    "inventors",
    "ipc_codes",
    "publication_date",
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

# method of use filter
MOU_FILTER = (
    "("
    "SELECT COUNT(1) FROM UNNEST(ipc_codes) AS ipc "
    f"JOIN UNNEST({METHOD_OF_USE_IPC_CODES}) AS mou_code "  # method of use
    "ON starts_with(ipc, mou_code)"
    ") = 0"
)


def __format_search_result(result: dict) -> PatentApplication:
    """
    Format a search result
    """
    dates = ["priority_date", "filing_date", "publication_date", "grant_date"]
    for date in dates:
        result[date] = (
            parse_date(str(result[date]), "%Y%m%d") if result.get(date) else None
        )
    return cast(PatentApplication, result)


def search(terms: list[str]) -> list[PatentApplication]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter, no method of use (TODO: too stringent?)

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
        "SELECT * "
        "FROM patents.entities, UNNEST(annotations) as annotation "
        f"WHERE annotation.term IN UNNEST({lower_terms}) "
        ") "
        f"SELECT {fields} "
        "FROM patents.applications AS applications "
        "JOIN filtered_entities AS entities "
        "ON applications.publication_number = entities.publication_number "
        "WHERE "
        f"priority_date > {max_priority_date} "  # min patent life
        f"AND {COM_FILTER} "
        f"AND {MOU_FILTER} "
        "limit 100"
    )
    results = select_from_bg(query)
    return [__format_search_result(result) for result in results]
