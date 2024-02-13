"""
Patent client
"""

import logging
import time
from typing import Optional
from prisma.partials import PatentDto
from prisma.models import Patent
from prisma.types import (
    PatentWhereInput,
    PatentWhereUniqueInput,
    PatentInclude,
    PatentOrderByInput,
    PatentScalarFieldKeys,
)

from clients.companies import get_financial_map
from clients.low_level.prisma import prisma_context
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

    async with prisma_context(300) as db:
        logger.info("AT PATENTS, with context")
        patents = await PatentDto.prisma(db).find_many(
            take, skip, where, cursor, include, order, distinct
        )
        logger.info("SUCCESSFULLY GOT PATENTS")

    logger.info(
        "Find took %s seconds (%s)", round(time.monotonic() - start, 2), len(patents)
    )

    ids = [p.id for p in patents]
    financial_map = await get_financial_map(ids, "patent_id")
    enriched_results = enrich_search_result(patents, financial_map)

    logger.info(
        "Find + enrichment took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(patents),
    )

    return enriched_results
