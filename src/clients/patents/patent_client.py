"""
Patent client
"""
from functools import partial
from typing import Sequence, cast
import logging

from clients import select_from_bg
from typings import PatentApplication

from .constants import COMPOSITION_OF_MATTER_IPC_CODES, RELEVANCY_THRESHOLD_MAP
from .formatting import format_search_result
from .utils import get_max_priority_date
from .types import RelevancyThreshold, TermResult

MIN_TERM_FREQUENCY = 100
MAX_SEARCH_RESULTS = 2000

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
    "classes",
    "compounds",
    # "cited_by",
    "country",
    "diseases",
    "drugs",
    "effects",
    "family_id",
    "genes",
    # "cpc_codes",
    # "embedding_v1 as embeddings",
    # "filing_date",
    # "grant_date",
    "inventors",
    "ipc_codes",
    # "matched_terms",
    # "matched_domains",
    "mechanisms",
    "proteins",
    "search_rank",
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


def get_term_query(domain: str, new_domain: str, threshold: float) -> str:
    """
    Returns a query for a given domain

    Args:
        domain (str): domain to query
        new_domain (str): new domain name
        threshold (float): threshold for search rank
    """
    return f"""
        ARRAY(SELECT a.term FROM UNNEST(a.annotations) as a
        where a.domain = '{domain}'
        and EXP(-annotation.character_offset_start * {DECAY_RATE}) > {threshold})
        as {new_domain}
    """


def search(
    terms: Sequence[str],
    min_patent_years: int = 10,
    relevancy_threshold: RelevancyThreshold = "high",
) -> Sequence[PatentApplication]:
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
        >>> search(['asthma', 'astrocytoma'])
    """
    lower_terms = [term.lower() for term in terms]
    fields = ",".join(SEARCH_RETURN_FIELDS)
    max_priority_date = get_max_priority_date(min_patent_years)
    threshold = RELEVANCY_THRESHOLD_MAP[relevancy_threshold]
    _get_term_query = partial(get_term_query, threshold=threshold)

    query = f"""
        WITH matches AS (
            SELECT
                a.publication_number as publication_number,
                annotation.term as matched_term,
                annotation.domain as matched_domain,
                EXP(-annotation.character_offset_start * {DECAY_RATE}) as search_rank, --- exp decay scaling; higher is better
                {_get_term_query('drugs', 'drugs')},
                {_get_term_query('compounds', 'compounds')},
                {_get_term_query('classes', 'classes')},
                {_get_term_query('diseases', 'diseases')},
                {_get_term_query('effects', 'effects')},
                {_get_term_query('humangenes', 'genes')},
                {_get_term_query('proteins', 'proteins')},
                {_get_term_query('mechanisms', 'mechanisms')},
            FROM patents.annotations a,
            UNNEST(a.annotations) as annotation
            WHERE annotation.term IN UNNEST({lower_terms})
        ),
        grouped_matches AS (
            SELECT
                publication_number,
                ARRAY_AGG(matched_term) as matched_terms,
                ARRAY_AGG(matched_domain) as matched_domains,
                AVG(search_rank) as search_rank,
                ANY_VALUE(drugs) as drugs,
                ANY_VALUE(compounds) as compounds,
                ANY_VALUE(classes) as classes,
                ANY_VALUE(diseases) as diseases,
                ANY_VALUE(effects) as effects,
                ANY_VALUE(genes) as genes,
                ANY_VALUE(mechanisms) as mechanisms,
                ANY_VALUE(proteins) as proteins,
            FROM matches
            GROUP BY publication_number
        )
        SELECT {fields}
        FROM patents.applications AS apps
        JOIN (
            SELECT *
            FROM grouped_matches
            WHERE ARRAY_LENGTH(matched_terms) = ARRAY_LENGTH({lower_terms})
        ) AS matched_pubs
        ON apps.publication_number = matched_pubs.publication_number
        WHERE
        priority_date > {max_priority_date}
        AND {COM_FILTER}
        AND search_rank > {threshold}
        ORDER BY search_rank DESC
        limit {MAX_SEARCH_RESULTS}
    """
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
