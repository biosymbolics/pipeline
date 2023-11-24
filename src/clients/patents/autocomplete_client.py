"""
Patent client
"""
import logging
import time
from typing import Literal, cast

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import APPLICATIONS_TABLE, TERMS_TABLE


from .types import AutocompleteMode, AutocompleteResult, TermResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


TERM_SEARCH = f"""
        SELECT DISTINCT ON (count, term) terms.term, count
        FROM {TERMS_TABLE}
        WHERE text_search @@ to_tsquery('english', %s)
        ORDER BY count DESC
    """

ID_SEARCH = f"""
        SELECT publication_number as term, 1 as count
        FROM {APPLICATIONS_TABLE}
        WHERE publication_number ~* concat(%s::text, '.*')
        ORDER BY publication_number DESC
    """


def _format_term(entity: TermResult) -> AutocompleteResult:
    return {"id": entity["term"], "label": f"{entity['term']} ({entity['count']})"}


def _autocomplete(
    string: str, query: str = TERM_SEARCH, limit: int = 25
) -> list[AutocompleteResult]:
    """
    Generates an autocomplete list for a given string and query

    Args:
        string (str): string to search for
        query (str, optional): query to run. Defaults to TERM_SEARCH.
        limit (int, optional): number of results to return. Defaults to 25.

    Returns: a list of matching terms
    """
    start = time.monotonic()

    search_sql = f"{' & '.join(string.split(' '))}:*"
    query = f"{query} limit {limit}"

    results = PsqlDatabaseClient().select(query, [search_sql])
    formatted = [_format_term(cast(TermResult, result)) for result in results]

    logger.info(
        "Autocomplete for string %s took %s seconds",
        string,
        round(time.monotonic() - start, 2),
    )

    return formatted


def autocomplete_terms(string: str, limit: int = 25) -> list[AutocompleteResult]:
    """
    Autocomplete terms
    """
    return _autocomplete(string, TERM_SEARCH, limit)


def autocomplete_id(string: str, limit: int = 25) -> list[AutocompleteResult]:
    """
    Autocomplete publication numbers
    """
    return _autocomplete(string, ID_SEARCH, limit)


def autocomplete(
    string: str, mode: AutocompleteMode, limit: int = 25
) -> list[AutocompleteResult]:
    """
    Autocomplete terms or ids for patents
    """
    if mode == "id":
        return autocomplete_id(string, limit)

    return autocomplete_terms(string, limit)
