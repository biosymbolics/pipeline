"""
Trials client
"""
from functools import partial
import logging
import time
from typing import Sequence
from prisma.types import (
    TrialWhereInput,
    TrialWhereInputRecursive1,
    TrialWhereInputRecursive2,
)
from pydash import flatten
from clients.documents.utils import get_where_clause

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context

from typings import QueryType, TrialSearchParams
from typings.documents.common import TermField
from typings.documents.trials import ScoredTrial
from utils.string import get_id

from .client import find_many

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000


async def _search(
    terms: Sequence[str],
    query_type: QueryType = "OR",
    term_fields: Sequence[TermField] = [
        TermField.canonical_name,
        TermField.instance_rollup,
    ],
    limit: int = MAX_SEARCH_RESULTS,
) -> list[ScoredTrial]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    where = get_where_clause(terms, term_fields, query_type, TrialWhereInput)

    async with prisma_context(300):
        trials = await find_many(
            where=where,
            include={
                "interventions": True,
                "indications": True,
                "outcomes": True,
                "sponsor": True,
            },
            take=limit,
        )

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(trials)
    )

    return trials


async def search(p: TrialSearchParams) -> list[ScoredTrial]:
    """
    Search trials by terms
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

    return await retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredTrial(**t) for t in storage_decoder(str_data)],
    )
