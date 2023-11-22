"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence, cast
from clients.companies.companies import get_company_map

from clients.low_level.boto3 import retrieve_with_cache_check
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    AGGREGATED_ANNOTATIONS_TABLE,
    ANNOTATIONS_TABLE,
    APPLICATIONS_TABLE,
    PATENT_TO_REGULATORY_APPROVAL_TABLE,
    REGULATORY_APPROVAL_TABLE,
    PATENT_TO_TRIAL_TABLE,
    TERMS_TABLE,
    TRIALS_TABLE,
)
from typings.patents import ScoredPatentApplication as PatentApplication
from utils.string import get_id

from .enrich import enrich_search_result
from .types import AutocompleteTerm, QueryPieces, QueryType, TermField, TermResult
from .utils import get_max_priority_date


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50

"""
Larger decay rates will result in more matches

Usage:
    EXP(-annotation.character_offset_start * {DECAY_RATE}) > {threshold})
"""
DECAY_RATE = 1 / 2000


SEARCH_RETURN_FIELDS = {
    "apps.publication_number": "publication_number",
    "apps.title": "title",
    "abstract": "abstract",
    "application_number": "application_number",
    "country": "country",
    "domains": "domains",
    "embeddings": "embeddings",
    "family_id": "family_id",
    "ipc_codes": "ipc_codes",
    "priority_date": "priority_date",
    "similar_patents": "similar_patents",
    "url": "url",
}

APPROVED_SEARCH_RETURN_FIELDS = {
    "approval_indications": "approval_indications",
    "brand_name": "brand_name",
    "generic_name": "generic_name",
}

TRIAL_RETURN_FIELDS = {
    "array_agg(distinct trials.nct_id)": "nct_ids",
    "(array_agg(trials.phase ORDER BY trials.start_date desc))[1]": "max_trial_phase",
    "(array_agg(trials.status ORDER BY trials.start_date desc))[1]": "last_trial_status",
    "(array_agg(trials.last_updated_date ORDER BY trials.start_date desc))[1]": "last_trial_update",
}

FIELDS: list[str] = [
    *[
        f"max({field}) as {new_field}"
        for field, new_field in {
            **SEARCH_RETURN_FIELDS,
            **APPROVED_SEARCH_RETURN_FIELDS,
        }.items()
    ],
    *[f"{field} as {new_field}" for field, new_field in TRIAL_RETURN_FIELDS.items()],
]


def __get_query_pieces(
    terms: list[str],
    query_type: QueryType,
    min_patent_years: int,
    is_exhaustive: bool = False,
) -> QueryPieces:
    """
    Helper to generate pieces of patent search query
    """
    is_id_search = any([t.startswith("WO-") for t in terms])
    lower_terms = [t.lower() for t in terms]

    # if ids, ignore most of the standard criteria
    if is_id_search:
        if not all([t.startswith("WO-") for t in terms]):
            raise ValueError("Cannot mix id and term search")

        return QueryPieces(
            fields=[*FIELDS, "1 as search_rank"],
            where=f"WHERE apps.publication_number = ANY(%s)",
            params=[terms],
        )

    # exp decay scaling for search terms; higher is better
    fields = [
        *FIELDS,
        f"AVG(EXP(-annotation.character_offset_start * {DECAY_RATE})) as search_rank",
    ]
    max_priority_date = get_max_priority_date(int(min_patent_years))
    date_criteria = f"priority_date > '{max_priority_date}'::date"

    comparison = "&&" if query_type == "OR" else "@>"
    term_criteria = f"search_terms {comparison} %s"

    if is_exhaustive:  # aka do tsquery search too
        join_char = " & " if query_type == "AND" else " | "
        ts_query_terms = (join_char).join(
            [" & ".join(t.split(" ")) for t in lower_terms]
        )

        return QueryPieces(
            fields=fields,
            where=f"""
                WHERE {date_criteria}
                AND ({term_criteria} OR text_search @@ to_tsquery('english', %s))
            """,
            params=[lower_terms, ts_query_terms],
        )

    return QueryPieces(
        fields=fields,
        where=f"WHERE {date_criteria} AND {term_criteria}",
        params=[lower_terms],
    )


def _search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    min_patent_years: int = 10,
    term_field: TermField = "terms",
    limit: int = MAX_SEARCH_RESULTS,
    is_exhaustive: bool = False,  # will search via tsquery too (slow)
) -> list[PatentApplication]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    qp = __get_query_pieces(terms, query_type, min_patent_years, is_exhaustive)

    query = f"""
        SELECT {", ".join(qp["fields"])},
        max(agg_annotations.{term_field}) as terms,
        (CASE
            WHEN max(approval_dates) IS NOT NULL AND ARRAY_LENGTH(max(approval_dates), 1) > 0
            THEN (max(approval_dates))[1]
            ELSE NULL END
        ) as approval_date,
        (CASE WHEN max(approval_dates) IS NOT NULL THEN True ELSE False END) as is_approved
        FROM {APPLICATIONS_TABLE} AS apps
        JOIN {AGGREGATED_ANNOTATIONS_TABLE} as agg_annotations ON (agg_annotations.publication_number = apps.publication_number)
        JOIN {ANNOTATIONS_TABLE} as annotation ON annotation.publication_number = apps.publication_number -- for search_rank
        LEFT JOIN {PATENT_TO_REGULATORY_APPROVAL_TABLE} p2a ON p2a.publication_number = ANY(apps.all_base_publication_numbers)
        LEFT JOIN {REGULATORY_APPROVAL_TABLE} approvals ON approvals.regulatory_application_number = p2a.regulatory_application_number
        LEFT JOIN {PATENT_TO_TRIAL_TABLE} a2t ON a2t.publication_number = apps.publication_number
        LEFT JOIN {TRIALS_TABLE} ON trials.nct_id = a2t.nct_id
        {qp["where"]}
        GROUP BY apps.publication_number
        ORDER BY priority_date desc
        LIMIT {limit}
    """

    results = PsqlDatabaseClient().select(query, qp["params"])

    company_map = get_company_map()
    enriched_results = enrich_search_result(results, company_map)

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(results)
    )

    return enriched_results


def search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    min_patent_years: int = 10,
    term_field: TermField = "terms",
    limit: int = MAX_SEARCH_RESULTS,
    skip_cache: bool = False,
    is_exhaustive: bool = False,
) -> list[PatentApplication]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        terms (Sequence[str]): list of terms to search for
        query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        min_patent_years (int, optional): minimum patent age in years. Defaults to 10.
        term_field (TermField, optional): which field to search on. Defaults to "terms".
                Other values are `instance_rollup` (which are rollup terms at a high level of specificity, e.g. "Aspirin 50mg" might have a rollup term of "Aspirin")
                and `category_rollup` (wherein "Aspirin 50mg" might have a rollup category of "NSAIDs")
        limit (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
        skip_cache (bool, optional): whether to skip cache. Defaults to False.
        is_exhaustive (bool, optional): whether to search via tsquery too (slow). Defaults to False.

    Returns: a list of matching patent applications

    Example:
    ```
    from clients.patents import patent_client
    patent_client.search(['asthma', 'astrocytoma'], skip_cache=True)
    ```
    """
    args = {
        "terms": terms,
        "query_type": query_type,
        "min_patent_years": min_patent_years,
        "term_field": term_field,
        "is_exhaustive": is_exhaustive,
    }
    key = get_id(args)
    search_partial = partial(_search, **args)

    if skip_cache:
        return search_partial(limit=limit)

    return retrieve_with_cache_check(search_partial, key=key, limit=limit)


def autocomplete_terms(string: str, limit: int = 25) -> list[AutocompleteTerm]:
    """
    Fetch all terms from table `terms`
    Sort by term, then by count.

    Args:
        string (str): string to search for

    Returns: a list of matching terms
    """
    start = time.time()

    def format_term(entity: TermResult) -> AutocompleteTerm:
        return {"id": entity["term"], "label": f"{entity['term']} ({entity['count']})"}

    search_sql = f"{' & '.join(string.split(' '))}:*"
    query = f"""
        SELECT DISTINCT ON (count, term) terms.term, count
        FROM {TERMS_TABLE}
        WHERE text_search @@ to_tsquery('english', %s)
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
