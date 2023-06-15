"""
Patent client
"""
from typing import Sequence
import logging

from clients import select_from_bg
from typings import PatentBasicInfo

from .constants import COMPOSITION_OF_MATTER_IPC_CODES
from .formatting import format_search_result
from .utils import get_max_priority_date
from .types import TermResult

MIN_TERM_FREQUENCY = 100
MAX_SEARCH_RESULTS = 2000

SEARCH_RETURN_FIELDS = [
    "applications.publication_number",
    "priority_date",
    "title",
    "abstract",
    # "application_kind",
    "application_number",
    "assignees",
    "compounds",
    # "cited_by",
    "country",
    "diseases",
    "effects",
    "family_id",
    "genes",
    # "cpc_codes",
    # "embedding_v1 as embeddings",
    # "filing_date",
    # "grant_date",
    "inventors",
    "ipc_codes",
    "matched_term",
    "matched_domain",
    "proteins",
    "search_rank",  # search rank
    # "publication_date",
    "ARRAY(SELECT s.publication_number FROM UNNEST(similar) as s where s.publication_number like 'WO%') as similar",  # limit to WO patents
    "top_terms",
    "url",
]

# composition of matter filter
COM_FILTER = f"""
    (
        SELECT COUNT(1) FROM UNNEST(ipc_codes) AS ipc
        JOIN UNNEST({COMPOSITION_OF_MATTER_IPC_CODES}) AS com_code
        ON starts_with(ipc, com_code)
    ) > 0
"""


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
    query = f"""
        WITH matches AS (
            SELECT
                a.publication_number as publication_number,
                annotation.term as matched_term,
                annotation.domain as matched_domain,
                EXP(-annotation.character_offset_start / 2000) as search_rank, --- exp decay scaling; higher is better
                ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a where a.domain = 'drugs') as compounds,
                ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a where a.domain = 'diseases') as diseases,
                ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a where a.domain = 'effects') as effects,
                ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a where a.domain = 'humangenes') as genes,
                ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a where a.domain = 'proteins') as proteins,
            FROM patents.annotations a,
            UNNEST(a.annotations) as annotation
            WHERE annotation.term IN UNNEST({lower_terms})
        )
        SELECT {fields}
        FROM patents.applications AS applications
        JOIN matches
        ON applications.publication_number = matches.publication_number
        WHERE
        priority_date > {max_priority_date} --- min patent life
        AND {COM_FILTER} --- composition of matter IPC codes
        ORDER BY search_rank DESC --- higher better; TODO: priority_date DESC
        limit {MAX_SEARCH_RESULTS}
    """
    results = select_from_bg(query)
    return format_search_result(results)


def __format_term(entity: TermResult) -> str:
    """
    Format an entity for autocomplete

    Args:
        entity (TermResult): entity to format
    """
    return f"{entity['term']} ({entity['count']})"


def autocomplete_terms(string: str) -> list[str]:
    """
    Fetch all terms from patents.terms
    Sort by term, then by count. Terms must have a count > MIN_TERM_FREQUENCY

    Args:
        string (str): string to search for

    Returns: a list of matching terms
    """
    query = f"""
        SELECT *
        FROM patents.terms
        WHERE term LIKE '%{string}%'
        AND count > {MIN_TERM_FREQUENCY}
        ORDER BY term ASC, count DESC
    """
    results = select_from_bg(query)
    return [__format_term(result) for result in results]
