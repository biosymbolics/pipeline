"""
Patent client
"""
from functools import partial
import logging
import os
import time
from typing import Sequence, cast
from pydash import compact

from clients.low_level.postgres import PsqlDatabaseClient
from constants.patents import COMPOSITION_OF_MATTER_IPC_CODES

from .constants import DOMAINS_OF_INTEREST, RELEVANCY_THRESHOLD_MAP
from .formatting import format_search_result
from .types import AutocompleteTerm, RelevancyThreshold, SearchResults, TermResult
from .utils import get_max_priority_date

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_TERM_FREQUENCY = 2
MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50

"""
Larger decay rates will result in more matches

Usage:
    EXP(-a.character_offset_start * {DECAY_RATE}) > {threshold})
"""
DECAY_RATE = 1 / 2000


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
    # "embeddings",
    # "grant_date",
    "inventors",
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
) -> SearchResults:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        terms (Sequence[str]): list of terms to search for

    Returns: a list of matching patent applications

    Example:
    ```
    import system; system.initialize()
    from clients.patents import patent_client
    patent_client.search(['asthma', 'astrocytoma'])
    ```
    """
    if not isinstance(terms, list):
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
        terms_sql = ", ".join([f"'{term}'" for term in terms])
        terms_count = 0  # enables an OR
        search_query = f"""
            SELECT
                publication_number,
                [] as matched_domains,
                [] as matched_terms,
                1 as search_rank
            from applications
            WHERE publication_number in ({terms_sql})
        """
    else:
        terms_sql = ", ".join([f"'{term.lower()}'" for term in terms])
        terms_count = len(terms)  # enables an AND
        search_query = f"""
            SELECT
                a.publication_number as publication_number,
                ARRAY_AGG(distinct a.domain) as matched_domains,
                ARRAY_AGG(DISTINCT a.term) as matched_terms,
                AVG(EXP(-a.character_offset_start * {DECAY_RATE})) as search_rank --- exp decay scaling; higher is better
            FROM annotations a
            WHERE lower(a.term) in ({terms_sql})
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
                domain,
                ARRAY_AGG(term) AS terms,
                ARRAY_AGG(domain) AS domains
            FROM annotations
            -- early filter for perf
            JOIN matches ON ({match_join})
            WHERE domain in ({", ".join([f"'{d}'" for d in DOMAINS_OF_INTEREST])})
            GROUP BY annotations.publication_number, domain
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

    logger.debug("Query: %s", query)
    print(query)
    results = PsqlDatabaseClient().select(query)
    return format_search_result(results)


def autocomplete_terms(string: str) -> list[AutocompleteTerm]:
    """
    Fetch all terms from table `terms`
    Sort by term, then by count. Terms must have a count > MIN_TERM_FREQUENCY

    Args:
        string (str): string to search for

    Returns: a list of matching terms
    """
    start = time.time()

    def format_term(entity: TermResult) -> AutocompleteTerm:
        return {"id": entity["term"], "label": f"{entity['term']} ({entity['count']})"}

    query = f"""
        SELECT *
        FROM terms
        WHERE term LIKE '%{string}%'
        AND count > {MIN_TERM_FREQUENCY}
        ORDER BY term ASC, count DESC
    """
    results = PsqlDatabaseClient().select(query)
    formatted = [format_term(cast(TermResult, result)) for result in results]

    logger.info(
        "Autocomplete for string %s took %s seconds",
        string,
        round(time.time() - start, 2),
    )

    return formatted
