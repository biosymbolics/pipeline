"""
Regulatory approvals client
"""
from functools import partial
import logging
import os
import time
from typing import Sequence
from prisma.models import RegulatoryApproval
from prisma.types import RegulatoryApprovalWhereInput

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import get_prisma_client

from typings import QueryType, ApprovalSearchParams
from utils.string import get_id


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_SEARCH_RESULTS = 2000


def get_where_clause(
    terms: Sequence[str],
    query_type: QueryType,
) -> RegulatoryApprovalWhereInput:
    lower_terms = [t.lower() for t in terms]
    where: RegulatoryApprovalWhereInput = {
        "OR": [
            {"interventions": {"some": {"canonical_name": {"in": lower_terms}}}},
            {"indications": {"some": {"canonical_name": {"in": lower_terms}}}},
        ]
    }

    return where


async def _search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[RegulatoryApproval]:
    """
    Search regulatory approvals by terms

    REPL:
    ```
    import asyncio
    from clients.approvals.search_client import _search
    asyncio.run(_search(["asthma"]))
    ```
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    where = get_where_clause(terms, query_type)

    async with get_prisma_client(300):
        approvals = await RegulatoryApproval.prisma().find_many(
            where=where,
            include={
                "interventions": True,
                "indications": True,
            },
            take=limit,
        )

    logger.info(
        "Search took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(approvals),
    )

    return approvals


async def search(p: ApprovalSearchParams) -> list[RegulatoryApproval]:
    """
    Search regulatory approvals by terms
    """
    args = {
        "terms": p.terms,
        "query_type": p.query_type,
    }
    key = get_id(
        {
            **args,
            "api": "approvals",
        }
    )
    search_partial = partial(_search, **args)

    if p.skip_cache == True:
        approvals = await search_partial(limit=p.limit)
        return approvals

    return retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [
            RegulatoryApproval(**a) for a in storage_decoder(str_data)
        ],
    )