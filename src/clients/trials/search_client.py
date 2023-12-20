"""
Trials client
"""
from functools import partial
import logging
import time
from typing import Sequence
from pydash import omit

from clients.low_level.boto3 import retrieve_with_cache_check
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import TRIALS_TABLE
from typings.trials import ScoredTrialSummary
from typings import QueryType, TrialSearchParams
from utils.sql import get_term_sql_query
from utils.string import get_id


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000


def _search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[ScoredTrialSummary]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    term_query = get_term_sql_query(terms, query_type)

    query = f"""
        SELECT *,
        ts_rank_cd(text_search, to_tsquery(%s)) AS score
        FROM {TRIALS_TABLE} as trials
        WHERE text_search @@ to_tsquery(%s)
        AND purpose in ('TREATMENT', 'BASIC_SCIENCE', 'PREVENTION')
        AND intervention_type='PHARMACOLOGICAL'
        ORDER BY score DESC
        LIMIT {limit}
    """

    results = PsqlDatabaseClient().select(query, [term_query, term_query])

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(results)
    )

    trials = [ScoredTrialSummary(**omit(r, ["text_search"])) for r in results]

    return trials


def search(p: TrialSearchParams) -> list[ScoredTrialSummary]:
    """
    Search trials by terms
    Filters on lowered, stemmed terms

    Usage:
    ```
    from clients.trials.search_client import search
    from typings import QueryType, TrialSearchParams
    search(TrialSearchParams(terms= ["asthma"], skip_cache=True))
    ```
    """
    args = {
        "terms": p.terms,
        "query_type": p.query_type,
    }
    key = get_id(args)
    search_partial = partial(_search, **args)

    if p.skip_cache == True:
        return search_partial(limit=p.limit)

    return retrieve_with_cache_check(search_partial, key=key, limit=p.limit)
