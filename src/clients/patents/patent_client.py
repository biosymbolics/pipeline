"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence, TypedDict, cast
from pydash import compact, flatten
from clients.low_level.boto3 import retrieve_with_cache_check

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import AGGREGATED_ANNOTATIONS_TABLE
from typings.patents import PatentApplication
from utils.string import get_id

from .formatting import format_search_result
from .types import AutocompleteTerm, TermResult
from .utils import get_max_priority_date

QueryPieces = TypedDict(
    "QueryPieces",
    {"match_condition": str, "where": str, "params": list},
)

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
    limit: int = MAX_SEARCH_RESULTS,
    is_randomized: bool = False,
    skip_cache: bool = False,
    is_exhaustive: bool = False,
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
        limit (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
        is_randomized (bool, optional): whether to randomize results. Defaults to False.
        skip_cache (bool, optional): whether to skip cache. Defaults to False.
        is_exhaustive (bool, optional): whether to search via tsquery too (slow). Defaults to False.

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
        "is_randomized": is_randomized,
        "is_exhaustive": is_exhaustive,
    }
    key = get_id(args)
    search_partial = partial(_search, **args)

    if skip_cache:
        return search_partial(limit=limit)

    return retrieve_with_cache_check(search_partial, key=key, limit=limit)


def __get_query_pieces(
    terms: list[str],
    domains: Sequence[str] | None,
    min_patent_years: int,
    limit: int,
    is_exhaustive: bool = False,
) -> QueryPieces:
    """
    Helper to generate pieces of patent search query
    """

    is_id_search = any([t.startswith("WO-") for t in terms])

    # if ids, ignore most of the standard criteria
    if is_id_search:
        if not all([t.startswith("WO-") for t in terms]):
            raise ValueError("Cannot mix id and term search")

        return {
            "match_condition": f"publication_number = any(%s)",
            "where": "",
            "params": [terms],
        }

    max_priority_date = get_max_priority_date(int(min_patent_years))
    base_params = {
        "where": f"""
            WHERE priority_date > '{max_priority_date}'::date
            ORDER BY randomizer desc, priority_date desc
            limit {limit}
        """,
        "match_condition": f"""
            terms @> %s -- terms contains all input terms
            {"AND domains @> %s" if domains else ""}
        """,
    }

    lower_terms = [t.lower() for t in terms]

    if is_exhaustive:  # aka do tsquery search too
        # AND all words in all supplied terms (TODO: this could obviously be more precise)
        ts_query_terms = (" & ").join(flatten([t.split(" ") for t in lower_terms]))

        return cast(
            QueryPieces,
            {
                **base_params,
                # text_search in match_condition so we can use JOIN instead of LEFT_JOIN
                "match_condition": f"AND ({base_params['match_condition']} OR text_search @@ to_tsquery('english', %s))",
                "params": compact([lower_terms, domains, ts_query_terms]),
            },
        )

    return cast(
        QueryPieces,
        {
            **base_params,
            "match_condition": f"AND {base_params['match_condition']}",
            "params": compact([lower_terms, domains]),
        },
    )


def _search(
    terms: Sequence[str],
    domains: Sequence[str] | None = None,
    min_patent_years: int = 10,
    limit: int = MAX_SEARCH_RESULTS,
    is_exhaustive: bool = False,  # will search via tsquery too (slow)
    is_randomized: bool = False,
) -> Sequence[PatentApplication]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    fields = compact(
        [
            *SEARCH_RETURN_FIELDS,
            *APPROVED_SEARCH_RETURN_FIELDS,
            "(CASE WHEN approval_date IS NOT NULL THEN 1 ELSE 0 END) * (random() - 0.9) as randomizer"
            if is_randomized
            else "1 as randomizer",  # for randomizing approved patents
        ]
    )

    qp = __get_query_pieces(terms, domains, min_patent_years, limit, is_exhaustive)

    query = f"""
        SELECT {", ".join(fields)}, terms, domains
        FROM applications AS apps
        JOIN {AGGREGATED_ANNOTATIONS_TABLE} as annotations ON (
            annotations.publication_number = apps.publication_number
            {qp["match_condition"]}
        )
        -- TODO: duplicates here
        LEFT JOIN patent_approvals approvals ON approvals.publication_number = ANY(apps.all_base_publication_numbers)
        {qp["where"]}
    """

    results = PsqlDatabaseClient().select(query, qp["params"])
    formatted_results = format_search_result(results)

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(results)
    )

    return formatted_results


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
        FROM terms
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
