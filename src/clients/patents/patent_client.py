"""
Patent client
"""
import json
from typing import Any, Mapping, Sequence, cast
import polars as pl
import logging

from clients import select_from_bg

from .constants import COMPOSITION_OF_MATTER_IPC_CODES, METHOD_OF_USE_IPC_CODES
from .utils import (
    clean_assignee,
    get_max_priority_date,
    get_patent_years,
    get_patent_attributes,
)
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

# method of use filter
MOU_FILTER = (
    "("
    "SELECT COUNT(1) FROM UNNEST(ipc_codes) AS ipc "
    f"JOIN UNNEST({METHOD_OF_USE_IPC_CODES}) AS mou_code "  # method of use
    "ON starts_with(ipc, mou_code)"
    ") = 0"
)


def __format_search_result(
    results: Sequence[dict[str, Any]]
) -> Sequence[PatentBasicInfo]:
    """
    Format a search result
    """
    df = pl.from_dicts(results)

    df = df.with_columns(
        pl.col("priority_date")
        .cast(str)
        .str.strptime(pl.Date, "%Y%m%d")
        .alias("priority_date"),
        pl.col("assignees").apply(
            lambda r: [clean_assignee(assignee) for assignee in r]
        ),
        pl.col("title").map(lambda t: get_patent_attributes(t)).alias("attributes"),
    )

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))

    return cast(Sequence[PatentBasicInfo], df.to_dicts())


def search(terms: Sequence[str]) -> Sequence[PatentBasicInfo]:
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
        "ORDER BY priority_date DESC "
        "limit 1000"
    )
    results = select_from_bg(query)
    return __format_search_result(results)
