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
from constants.patents import COMPOSITION_OF_MATTER_IPC_CODES
from typings.patents import PatentApplication
from utils.string import get_id

from .constants import DOMAINS_OF_INTEREST, RELEVANCY_THRESHOLD_MAP
from .formatting import format_search_result
from .types import (
    AutocompleteTerm,
    RelevancyThreshold,
    TermResult,
)
from .utils import get_max_priority_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50

"""
Larger decay rates will result in more matches

Usage:
    EXP(-a.character_offset_start * {DECAY_RATE}) > {threshold})
"""
DECAY_RATE = 1 / 20000


SEARCH_RETURN_FIELDS = [
    "apps.publication_number",
    "priority_date",
    "title",
    "abstract",
    # "application_kind",
    "application_number",
    # "assignees",
    "country",
    "family_id",
    "filing_date",
    "embeddings",
    "grant_date",
    # "inventors",
    "ipc_codes",
    "search_rank",
    # 'ARRAY(SELECT s.publication_number FROM "similar" as s where s.publication_number like \'WO%\') as "similar"',  # limit to WO patents
    '"similar"',
    "url",
]

APPROVED_SERACH_RETURN_FIELDS = [
    "brand_name",
    "generic_name",
    "approval_date",
    "patent_indication as indication",
]

# composition of matter filter
# TODO: Keep? it is slow
COM_FILTER = f"""
    (
        SELECT COUNT(1) FROM UNNEST(ipc_codes) AS ipc
        JOIN unnest(ARRAY{COMPOSITION_OF_MATTER_IPC_CODES}) AS com_code
        ON starts_with(ipc, com_code)
    ) > 0
"""


def search(
    terms: Sequence[str],
    fetch_approval: bool = False,
    min_patent_years: int = 10,
    relevancy_threshold: RelevancyThreshold = "high",
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
        fetch_approval (bool, optional): whether to fetch approval info. Defaults to False.
        min_patent_years (int, optional): minimum patent age in years. Defaults to 10.
        relevancy_threshold (RelevancyThreshold, optional): relevancy threshold. Defaults to "high".
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
        "fetch_approval": fetch_approval,
        "min_patent_years": min_patent_years,
        "relevancy_threshold": relevancy_threshold,
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
    fetch_approval: bool = False,
    min_patent_years: int = 10,
    relevancy_threshold: RelevancyThreshold = "high",
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

    threshold = RELEVANCY_THRESHOLD_MAP[relevancy_threshold]
    max_priority_date = get_max_priority_date(int(min_patent_years))

    fields = ", ".join(
        compact(
            [
                *SEARCH_RETURN_FIELDS,
                *(APPROVED_SERACH_RETURN_FIELDS if fetch_approval else []),
                "(CASE WHEN approval_date IS NOT NULL THEN 1 ELSE 0 END) * (random() - 0.9) as randomizer"
                if is_randomized and fetch_approval
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
                1 as search_rank
            from applications
            WHERE publication_number = any(%s)
        """
    else:
        _terms = [term.lower() for term in terms]
        terms_count = len(terms)  # enables an AND
        search_query = f"""
            SELECT
                a.publication_number as publication_number,
                ARRAY_AGG(distinct a.domain) as matched_domains,
                ARRAY_AGG(DISTINCT a.term) as matched_terms,
                AVG(EXP(-a.character_offset_start * {DECAY_RATE})) as search_rank --- exp decay scaling; higher is better
            FROM annotations a
            WHERE lower(a.term) = any(%s)
            GROUP BY a.publication_number
        """

    match_join = f"""
        annotations.publication_number = matches.publication_number
        AND
        ARRAY_LENGTH(matched_terms, 1) >= {terms_count}
    """

    select_query = f"""
        WITH matches AS ({search_query}),
        annotations AS (
            SELECT
                annotations.publication_number,
                ARRAY_AGG(term) AS terms,
                ARRAY_AGG(domain) AS domains
            FROM annotations
            -- early filter for perf
            JOIN matches ON ({match_join})
            WHERE domain in ({", ".join([f"'{d}'" for d in DOMAINS_OF_INTEREST])})
            GROUP BY annotations.publication_number
        )
        SELECT {fields}, terms, domains
        FROM applications AS apps
        JOIN annotations on annotations.publication_number = apps.publication_number
        JOIN matches ON ({match_join})
    """

    if fetch_approval:
        select_query += """
            LEFT JOIN patent_approvals approvals
            ON approvals.publication_number = ANY(apps.all_base_publication_numbers)
        """

    where = f"""
        WHERE priority_date > '{max_priority_date}'::date
        AND search_rank > {threshold}
        ORDER BY randomizer desc, search_rank DESC
        limit {max_results}
    """

    query = select_query + where
    results = PsqlDatabaseClient().select(query, [_terms])

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
        SELECT *
        FROM terms
        WHERE term ~* %s
        ORDER BY count DESC
        limit {limit}
    """
    results = PsqlDatabaseClient().select(query, [search_sql])
    formatted = [format_term(cast(TermResult, result)) for result in results]

    logger.info(
        "Autocomplete for string %s took %s seconds",
        string,
        round(time.time() - start, 2),
    )

    return formatted
