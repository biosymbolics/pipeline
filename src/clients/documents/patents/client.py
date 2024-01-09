"""
Patent client
"""
import inspect
import logging
import time
from typing import Optional
from prisma.actions import PatentActions
from prisma.models import Patent
from prisma.types import (
    PatentWhereInput,
    PatentWhereUniqueInput,
    PatentInclude,
    PatentOrderByInput,
    PatentScalarFieldKeys,
)

from clients.companies import get_financial_map
from clients.low_level.prisma import get_prisma_client
from typings import ScoredPatent

from .enrich import enrich_search_result


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def find_many(
    take: Optional[int] = None,
    skip: Optional[int] = None,
    where: Optional[PatentWhereInput] = None,
    cursor: Optional[PatentWhereUniqueInput] = None,
    include: Optional[PatentInclude] = None,
    order: Optional[PatentOrderByInput | list[PatentOrderByInput]] = None,
    distinct: Optional[list[PatentScalarFieldKeys]] = None,
) -> list[ScoredPatent]:
    """
    Find patents
    ```
    """
    start = time.monotonic()

    patents = await Patent.prisma().find_many(
        take, skip, where, cursor, include, order, distinct
    )

    ids = [p.id for p in patents]
    financial_map = await get_financial_map(ids, "assignee_patent_id")
    enriched_results = enrich_search_result(patents, financial_map)

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(patents)
    )

    return enriched_results
