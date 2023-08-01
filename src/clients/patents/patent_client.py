"""
Patent client
"""
from functools import partial
from typing import Sequence, Union, cast
import logging

from pydash import compact

from clients import select_from_bg
from typings import ApprovedPatentApplication, PatentApplication

from .constants import COMPOSITION_OF_MATTER_IPC_CODES, RELEVANCY_THRESHOLD_MAP
from .formatting import format_search_result
from .utils import get_max_priority_date
from .types import RelevancyThreshold, TermResult

MIN_TERM_FREQUENCY = 20
MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50

"""
Larger decay rates will result in more matches

Usage:
    EXP(-annotation.character_offset_start * {DECAY_RATE}) > {threshold})
"""
DECAY_RATE = 1 / 2000

SEARCH_RETURN_FIELDS = [
    "apps.publication_number",
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
    "search_rank",
    # "publication_date",
    "ARRAY(SELECT s.publication_number FROM UNNEST(similar) as s where s.publication_number like 'WO%') as similar",  # limit to WO patents
    # "top_terms",
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
        JOIN UNNEST({COMPOSITION_OF_MATTER_IPC_CODES}) AS com_code
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
            SELECT distinct a.term FROM UNNEST(a.annotations) as a
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
) -> Union[Sequence[PatentApplication], Sequence[ApprovedPatentApplication]]:
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
    lower_terms = [term.lower() for term in terms]
    threshold = RELEVANCY_THRESHOLD_MAP[relevancy_threshold]
    max_priority_date = get_max_priority_date(min_patent_years)
    _get_term_query = partial(get_term_query, threshold=threshold)
    fields = ",".join(
        compact(
            [
                *SEARCH_RETURN_FIELDS,
                *(APPROVED_SERACH_RETURN_FIELDS if fetch_approval else []),
                _get_term_query("compounds", "compounds"),
                _get_term_query("diseases", "diseases"),
                _get_term_query("humangenes", "genes"),
                _get_term_query("mechanisms", "mechanisms"),
                "(CASE WHEN approval_date IS NOT NULL THEN 1 ELSE 0 END) * (RAND() - 0.9) as randomizer"
                if fetch_approval
                else None,  # for randomizing approved patents
            ]
        )
    )

    select_query = f"""
        WITH matches AS (
            SELECT
                a.publication_number as publication_number,
                ARRAY_AGG(
                    DISTINCT
                    CASE
                        WHEN annotation.term IN UNNEST({lower_terms}) THEN annotation.term
                        WHEN lower(apps.publication_number) IN UNNEST({lower_terms}) THEN apps.publication_number
                        ELSE ""
                    END
                ) as matched_terms,
                ARRAY_AGG(distinct annotation.domain) as matched_domains,
                AVG(
                    CASE
                        WHEN annotation.term IN UNNEST({lower_terms}) THEN EXP(-annotation.character_offset_start * {DECAY_RATE})
                        WHEN lower(apps.publication_number) IN UNNEST({lower_terms}) THEN 1.0
                        ELSE 0
                    END
                ) as search_rank, --- exp decay scaling; higher is better
            FROM patents.annotations a,
            patents.applications AS apps,
            UNNEST(a.annotations) as annotation
            WHERE apps.publication_number = a.publication_number
            AND (
                lower(annotation.term) IN UNNEST({lower_terms})
                OR
                lower(apps.publication_number) IN UNNEST({lower_terms})
            )
            GROUP BY publication_number
        )
        SELECT {fields}
        FROM patents.applications AS apps
        JOIN patents.annotations a on a.publication_number = apps.publication_number
        JOIN matches ON (
            apps.publication_number = matches.publication_number
            and
            ARRAY_LENGTH(matched_terms) = ARRAY_LENGTH({lower_terms})
        )
    """

    if fetch_approval:
        select_query += """
            LEFT JOIN `patents.patent_approvals` approvals
            ON approvals.publication_number in unnest(apps.all_base_publication_numbers)
        """

    where = f"""
        WHERE
        priority_date > {max_priority_date}
        AND {COM_FILTER}
        AND search_rank > {threshold}
        ORDER BY {"randomizer desc, " if fetch_approval else ""} search_rank DESC
        limit {max_results}
    """

    query = select_query + where
    results = select_from_bg(query)
    return format_search_result(results)


def autocomplete_terms(string: str) -> list[str]:
    """
    Fetch all terms from patents.terms
    Sort by term, then by count. Terms must have a count > MIN_TERM_FREQUENCY

    Args:
        string (str): string to search for

    Returns: a list of matching terms
    """

    def format_term(entity: TermResult) -> str:
        return f"{entity['term']} ({entity['count']})"

    query = f"""
        SELECT *
        FROM patents.terms
        WHERE term LIKE '%{string}%'
        AND count > {MIN_TERM_FREQUENCY}
        ORDER BY term ASC, count DESC
    """
    results = select_from_bg(query)
    return [format_term(cast(TermResult, result)) for result in results]
