"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence, cast
from pydash import compact
from clients.low_level.boto3 import retrieve_with_cache_check

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import AGGREGATED_ANNOTATIONS_TABLE
from typings.patents import PatentApplication
from utils.string import get_id

from .formatting import format_search_result
from .types import AutocompleteTerm, TermResult
from .utils import get_max_priority_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50


SEARCH_RETURN_FIELDS = [
    "apps.publication_number",
    "priority_date",
    "abstract",
    # "application_kind",
    "application_number",
    "country",
    "embeddings",
    "explanation",
    "family_id",
    # "filing_date",
    # "grant_date",
    # "publication_date",
    "ipc_codes",
    '"similar"',
    "title",
    "url",
]

APPROVED_SEARCH_RETURN_FIELDS = [
    "approval_date",
    "brand_name",
    "generic_name",
    "(CASE WHEN approval_date IS NOT NULL THEN True ELSE False END) as is_approved",
    "patent_indication as indication",
]


def search(
    terms: Sequence[str],
    domains: Sequence[str] | None = None,
    min_patent_years: int = 10,
    max_results: int = MAX_SEARCH_RESULTS,
    is_randomized: bool = False,
    skip_cache: bool = False,
) -> Sequence[PatentApplication]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        terms (Sequence[str]): list of terms to search for
        domains (Sequence[str], optional): list of domains to filter on. Defaults to None.
        min_patent_years (int, optional): minimum patent age in years. Defaults to 10.
        max_results (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
        is_randomized (bool, optional): whether to randomize results. Defaults to False.
        skip_cache (bool, optional): whether to skip cache. Defaults to False.

    Returns: a list of matching patent applications

    Example:
    ```
    from clients.patents import patent_client
    patent_client.search(['asthma', 'astrocytoma'])
    ```
    """
    args = {
        "terms": terms,
        "domains": domains,
        "min_patent_years": min_patent_years,
        "max_results": max_results,
        "is_randomized": is_randomized,
    }
    key = get_id(args)
    search_partial = partial(_search, **args)

    if skip_cache:
        return search_partial()

    return retrieve_with_cache_check(search_partial, key=key)


def _search(
    terms: Sequence[str],
    domains: Sequence[str] | None = None,
    min_patent_years: int = 10,
    max_results: int = MAX_SEARCH_RESULTS,
    is_randomized: bool = False,
) -> Sequence[PatentApplication]:
    """
    Search patents by terms
    """
    start = time.time()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    # only checks for global patents
    is_id_search = any([t.startswith("WO-") for t in terms])

    if is_id_search and not all([t.startswith("WO-") for t in terms]):
        raise ValueError("Cannot mix id and term search")

    max_priority_date = get_max_priority_date(int(min_patent_years))

    fields = ", ".join(
        compact(
            [
                *SEARCH_RETURN_FIELDS,
                *APPROVED_SEARCH_RETURN_FIELDS,
                "(CASE WHEN approval_date IS NOT NULL THEN 1 ELSE 0 END) * (random() - 0.9) as randomizer"
                if is_randomized
                else "1 as randomizer",  # for randomizing approved patents
            ]
        )
    )

    if is_id_search:
        _terms = terms
        terms_count = 0  # enables an OR
        search_query = f"""
            SELECT
                publication_number,
                ARRAY[]::TEXT[] as matched_domains,
                ARRAY[]::TEXT[] as matched_terms,
            from applications
            WHERE publication_number = any(%s)
        """
    else:
        _terms = [term.lower() for term in terms]
        terms_count = len(terms)  # enables an AND
        domain_where = f"AND domain = any(%s)" if domains else ""
        search_query = f"""
            SELECT
                a.publication_number as publication_number,
                ARRAY_AGG(distinct a.domain) as matched_domains,
                ARRAY_AGG(DISTINCT a.term) as matched_terms
            FROM annotations a
            WHERE lower(a.term) = any(%s)
            {domain_where}
            GROUP BY a.publication_number
        """

    # TODO: duplicates on approvals
    select_query = f"""
        WITH matches AS ({search_query})
        SELECT {fields}, terms, domains
        FROM applications AS apps
        JOIN matches ON (
            apps.publication_number = matches.publication_number
            AND (
                coalesce(ARRAY_LENGTH(matched_terms, 1), 0) >= {terms_count}
                OR
                textsearch @@ to_tsquery('english', '{(" & ").join(_terms)}') # full text search alernative
            )
        )
        JOIN {AGGREGATED_ANNOTATIONS_TABLE} as annotations ON annotations.publication_number = apps.publication_number
        LEFT JOIN patent_approvals approvals ON approvals.publication_number = ANY(apps.all_base_publication_numbers)
    """

    if is_id_search:
        # don't constrain what's returned for id-only
        where = ""
    else:
        where = f"""
            WHERE priority_date > '{max_priority_date}'::date
            ORDER BY randomizer desc
            limit {max_results}
        """

    query = select_query + where
    results = PsqlDatabaseClient().select(query, compact([_terms, domains]))

    logger.debug("Patent search query: %s", query)
    logger.info(
        "Search took %s seconds (%s)", round(time.time() - start, 2), len(results)
    )

    return format_search_result(results)


def autocomplete_terms(string: str, limit: int = 25) -> list[AutocompleteTerm]:
    """
    Fetch all terms from table `terms`
    Sort by term, then by count.

    Args:
        string (str): string to search for

    Returns: a list of matching terms

    TODO: tsvector
    """
    start = time.time()

    def format_term(entity: TermResult) -> AutocompleteTerm:
        return {"id": entity["term"], "label": f"{entity['term']} ({entity['count']})"}

    search_sql = f".*{string}.*"
    query = f"""
        SELECT DISTINCT ON (count, term) terms.term, count
        FROM terms
        LEFT JOIN synonym_map ON terms.term = synonym_map.synonym
        WHERE (
            terms.term ~* %s
            OR
            synonym ~* %s
        )
        ORDER BY count DESC
        limit {limit}
    """
    results = PsqlDatabaseClient().select(query, [search_sql, search_sql])
    formatted = [format_term(cast(TermResult, result)) for result in results]

    logger.info(
        "Autocomplete for string %s took %s seconds",
        string,
        round(time.time() - start, 2),
    )

    return formatted
