"""
Approvals client
"""
import logging
import time
from typing import Optional
from prisma.models import RegulatoryApproval
from prisma.types import (
    RegulatoryApprovalWhereInput,
    RegulatoryApprovalWhereUniqueInput,
    RegulatoryApprovalInclude,
    RegulatoryApprovalOrderByInput,
    RegulatoryApprovalScalarFieldKeys,
)

from clients.low_level.prisma import get_prisma_client
from typings import ScoredRegulatoryApproval


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def find_many(
    take: Optional[int] = None,
    skip: Optional[int] = None,
    where: Optional[RegulatoryApprovalWhereInput] = None,
    cursor: Optional[RegulatoryApprovalWhereUniqueInput] = None,
    include: Optional[RegulatoryApprovalInclude] = None,
    order: Optional[
        RegulatoryApprovalOrderByInput | list[RegulatoryApprovalOrderByInput]
    ] = None,
    distinct: Optional[list[RegulatoryApprovalScalarFieldKeys]] = None,
) -> list[ScoredRegulatoryApproval]:
    """
    Find approvals
    ```
    """
    start = time.monotonic()

    approvals = await RegulatoryApproval.prisma().find_many(
        take, skip, where, cursor, include, order, distinct
    )

    logger.info(
        "Search took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(approvals),
    )

    return [ScoredRegulatoryApproval(**a.__dict__) for a in approvals]
