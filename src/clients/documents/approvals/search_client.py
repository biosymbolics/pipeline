"""
Regulatory approvals client
"""
from functools import partial
import logging
import time
from typing import Sequence
from prisma.types import RegulatoryApprovalWhereInput

from clients.documents.utils import get_where_clause
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import (
    QueryType,
    RegulatoryApprovalSearchParams,
    ScoredRegulatoryApproval,
    TermField,
)
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
) -> list[ScoredRegulatoryApproval]:
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

    where = get_where_clause(
        terms, term_fields, query_type, RegulatoryApprovalWhereInput
    )

    approvals = await find_many(
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


async def search(p: RegulatoryApprovalSearchParams) -> list[ScoredRegulatoryApproval]:
    """
    Search regulatory approvals by terms
    """
    args = {
        "terms": p.terms,
        "query_type": p.query_type,
        "include": p.include,
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

    return await retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [
            ScoredRegulatoryApproval(**a) for a in storage_decoder(str_data)
        ],
    )
