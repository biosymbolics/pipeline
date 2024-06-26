"""
Regulatory approvals client
"""

import logging
import re
from typing import Sequence
from prisma.types import RegulatoryApprovalWhereInput

from clients.documents.utils import (
    get_matching_doc_ids,
    get_search_clause,
)
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import (
    RegulatoryApprovalSearchParams,
    ScoredRegulatoryApproval,
)
from typings.client import DocumentSearchCriteria
from typings.documents.common import DocType
from utils.string import get_id

from .approvals_client import find_many


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROVAL_ID_RE = re.compile("^[0-9]{4,5}-[0-9]{3,4}$")


def get_where_clause(
    p: DocumentSearchCriteria,
    term_ids: Sequence[str] | None = None,
    description_ids: Sequence[str] | None = None,
) -> RegulatoryApprovalWhereInput:
    """
    Get where clause for regulatory approvals
    """
    is_id_search = any([APPROVAL_ID_RE.match(t) for t in p.terms])

    if is_id_search and any([not APPROVAL_ID_RE.match(t) for t in p.terms]):
        raise ValueError("ID search; all terms must be XXXXX?-XXXX?")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    return get_search_clause(
        DocType.regulatory_approval,
        p,
        term_ids,
        description_ids,
        return_type=RegulatoryApprovalWhereInput,
    )


async def search(
    params: RegulatoryApprovalSearchParams,
) -> list[ScoredRegulatoryApproval]:
    """
    Search regulatory approvals by terms
    """
    p = RegulatoryApprovalSearchParams.parse(params)
    search_criteria = DocumentSearchCriteria.parse(p)

    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "approvals",
        }
    )

    async def _search(limit: int):
        term_match_ids, vector_match_ids = await get_matching_doc_ids(
            p,
            [DocType.trial],
        )

        where = get_where_clause(search_criteria, term_match_ids, vector_match_ids)
        return await find_many(where=where, include=p.include, take=limit)

    if p.skip_cache == True:
        approvals = await _search(limit=p.limit)
        return approvals

    return await retrieve_with_cache_check(
        _search,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [
            ScoredRegulatoryApproval(**a) for a in storage_decoder(str_data)
        ],
    )
