"""
Regulatory approvals client
"""
from functools import partial
import logging
import os
import time
from typing import Sequence
from prisma import Prisma
from prisma.models import RegulatoryApproval

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from constants.core import DATABASE_URL, REGULATORY_APPROVAL_TABLE

from typings import QueryType, ApprovalSearchParams
from utils.sql import get_term_sql_query
from utils.string import get_id


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["DATABASE_URL"] = DATABASE_URL or ""

MAX_SEARCH_RESULTS = 2000

FIELDS = [
    # "applicant",
    "application_type",
    "approval_date",
    # "application_number",
    "brand_name",
    "generic_name",
    "indications",
    "label_url",
    "ndc_code",
    "pharmacologic_class",
    "pharmacologic_classes",
    "regulatory_agency",
]


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

    term_query = get_term_sql_query(terms, query_type)

    # prisma py doesn't yet support to_tsquery
    query = f"""
        SELECT *,
        ts_rank_cd(search, to_tsquery($1)) AS score
        FROM {REGULATORY_APPROVAL_TABLE} as approvals
        WHERE search @@ to_tsquery($2)
        AND approval_date is not null
        ORDER BY score DESC
        LIMIT {limit}
    """

    async with Prisma(auto_register=True, http={"timeout": 300}):
        approvals = await RegulatoryApproval.prisma().query_raw(
            query, term_query, term_query
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
