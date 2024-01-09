"""
Trials client
"""
from functools import partial
import logging
import time
from typing import Sequence
from prisma.models import Trial
from prisma.types import TrialWhereInput

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import get_prisma_client

from typings import QueryType, TrialSearchParams
from utils.string import get_id


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000


def get_where_clause(
    terms: Sequence[str],
    query_type: QueryType,
) -> TrialWhereInput:
    lower_terms = [t.lower() for t in terms]
    where: TrialWhereInput = {
        "OR": [
            {"interventions": {"some": {"canonical_name": {"in": lower_terms}}}},
            {"indications": {"some": {"canonical_name": {"in": lower_terms}}}},
            {"sponsor": {"is": {"canonical_name": {"in": lower_terms}}}},
        ]
    }

    return where


async def _search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[Trial]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    where = get_where_clause(terms, query_type)

    async with get_prisma_client(300):
        trials = await Trial.prisma().find_many(
            where=where,
            include={
                "interventions": True,
                "indications": True,
                "sponsor": True,
            },
            take=limit,
        )

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(trials)
    )

    return trials


async def search(p: TrialSearchParams) -> list[Trial]:
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
    key = get_id(
        {
            **args,
            "api": "trials",
        }
    )
    search_partial = partial(_search, **args)

    if p.skip_cache == True:
        trials = await search_partial(limit=p.limit)
        return trials

    return retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [Trial(**t) for t in storage_decoder(str_data)],
    )
