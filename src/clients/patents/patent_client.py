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

from .constants import RELEVANCY_THRESHOLD_MAP
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


def get_term_query(domain: str, new_domain: str, threshold: float) -> str:
    """
    Returns a query for a given domain

    Args:
        domain (str): domain to query
        new_domain (str): new domain name
        threshold (float): threshold for search rank
    """
    return f"""
        ARRAY(
            SELECT distinct a.term FROM annotations a
            where a.domain = '{domain}'
            and length(a.term) > 1
            and EXP(-a.character_offset_start * {DECAY_RATE}) > {threshold}
            limit {MAX_ARRAY_LENGTH}
        )
        as {new_domain}
    """


def search(
    terms: Sequence[str],
    fetch_approval: bool = False,
    min_patent_years: int = 10,
    relevancy_threshold: RelevancyThreshold = "high",
    max_results: int = MAX_SEARCH_RESULTS,
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

    terms_count = len(terms)
    lower_terms = ", ".join([f"'{term.lower()}'" for term in terms])
    threshold = RELEVANCY_THRESHOLD_MAP[relevancy_threshold]
    max_priority_date = get_max_priority_date(int(min_patent_years))
    _get_term_query = partial(get_term_query, threshold=threshold)
    fields = ",".join(
        compact(
            [
                *SEARCH_RETURN_FIELDS,
                *(APPROVED_SERACH_RETURN_FIELDS if fetch_approval else []),
                _get_term_query("assignee", "assignees"),
                _get_term_query("attributes", "attributes"),
                _get_term_query("compounds", "compounds"),
                _get_term_query("diseases", "diseases"),
                _get_term_query("humangenes", "genes"),
                _get_term_query("mechanisms", "mechanisms"),
                "(CASE WHEN approval_date IS NOT NULL THEN 1 ELSE 0 END) * (random() - 0.9) as randomizer"
                if fetch_approval
                else None,  # for randomizing approved patents
            ]
        )
    )

    select_query = f"""
        WITH matches AS (
            SELECT
                apps.publication_number as publication_number,
                ARRAY_AGG(
                    DISTINCT
                    CASE
                        WHEN a.term IN ({lower_terms}) THEN a.term
                        WHEN lower(apps.publication_number) IN({lower_terms}) THEN apps.publication_number
                        ELSE ''::text
                    END
                ) as matched_terms,
                ARRAY_AGG(distinct a.domain) as matched_domains,
                AVG(
                    CASE
                        WHEN a.term IN ({lower_terms}) THEN EXP(-a.character_offset_start * {DECAY_RATE})
                        WHEN lower(apps.publication_number) IN ({lower_terms}) THEN 1.0
                        ELSE 0
                    END
                ) as search_rank --- exp decay scaling; higher is better
            FROM
                annotations a,
                applications AS apps
            WHERE apps.publication_number = a.publication_number
            AND (
                lower(a.term) IN ({lower_terms})
                OR
                lower(apps.publication_number) IN ({lower_terms})
            )
            GROUP BY apps.publication_number
        )
        SELECT {fields}
        FROM applications AS apps
        JOIN annotations a on a.publication_number = apps.publication_number
        JOIN matches ON (
            apps.publication_number = matches.publication_number
            and
            ARRAY_LENGTH(matched_terms, 1) = {terms_count}
        )
    """

    if fetch_approval:
        select_query += """
            LEFT JOIN patent_approvals approvals
            ON approvals.publication_number = ANY(apps.all_base_publication_numbers)
        """

    where = f"""
        WHERE
        priority_date > '{max_priority_date}'::date
        AND {COM_FILTER}
        AND search_rank > {threshold}
        ORDER BY {"randomizer desc, " if fetch_approval else ""}search_rank DESC
        limit {max_results}
    """

    query = select_query + where
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
