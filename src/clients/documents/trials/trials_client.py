"""
Approvals client
"""

import logging
import time
from typing import Optional
from prisma.partials import TrialDto
from prisma.types import (
    TrialWhereInput,
    TrialWhereUniqueInput,
    TrialInclude,
    TrialOrderByInput,
    TrialScalarFieldKeys,
)

from clients.low_level.prisma import prisma_context
from typings import ScoredTrial


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def find_many(
    take: Optional[int] = None,
    skip: Optional[int] = None,
    where: Optional[TrialWhereInput] = None,
    cursor: Optional[TrialWhereUniqueInput] = None,
    include: Optional[TrialInclude] = None,
    order: Optional[TrialOrderByInput | list[TrialOrderByInput]] = None,
    distinct: Optional[list[TrialScalarFieldKeys]] = None,
) -> list[ScoredTrial]:
    """
    Find trials
    """
    start = time.monotonic()

    async with prisma_context(300) as db:
        trials = await TrialDto.prisma(db).find_many(
            take, skip, where, cursor, include, order, distinct
        )

    logger.info(
        "Search took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(trials),
    )

    return [ScoredTrial(**t.__dict__) for t in trials]
