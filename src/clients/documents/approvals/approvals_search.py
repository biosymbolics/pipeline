"""
Regulatory approvals client
"""

from datetime import datetime
import logging
import re
from prisma.types import RegulatoryApprovalWhereInput

from clients.documents.utils import get_term_clause
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import (
    RegulatoryApprovalSearchParams,
    ScoredRegulatoryApproval,
)
from typings.client import (
    DocumentSearchCriteria,
    DocumentSearchParams,
)
from utils.string import get_id

from .approvals_client import find_many


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPROVAL_ID_RE = re.compile("^[0-9]{4,5}-[0-9]{3,4}$")


def get_where_clause(p: DocumentSearchCriteria) -> RegulatoryApprovalWhereInput:
    """
    Get where clause for regulatory approvals
    """
    is_id_search = any([APPROVAL_ID_RE.match(t) for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not APPROVAL_ID_RE.match(t) for t in p.terms]):
        raise ValueError("ID search; all terms must be XXXXX?-XXXX?")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    term_clause = get_term_clause(p, RegulatoryApprovalWhereInput)

    where: RegulatoryApprovalWhereInput = {
        **term_clause,
        "approval_date": {
            "gte": datetime(p.start_year, 1, 1),
            "lte": datetime(p.end_year, 1, 1),
        },
    }

    return where


async def search(
    params: DocumentSearchParams | RegulatoryApprovalSearchParams,
) -> list[ScoredRegulatoryApproval]:
    """
    Search regulatory approvals by terms

    Args:
        p.terms (Sequence[str]): list of terms to search for
        p.include (RegulatoryApprovalInclude, optional): whether to include assignees, inventors, interventions, indications. Defaults to DEFAULT_PATENT_INCLUDE.
        p.start_year (int, optional): minimum priority date year. Defaults to DEFAULT_START_YEAR.
        p.end_year (int, optional): maximum priority date year. Defaults to DEFAULT_END_YEAR.
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.term_fields (Sequence[TermField], optional): which fields to search for terms in. Defaults to DEFAULT_TERM_FIELDS.
        p.limit (int, optional): max results to return.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.
    """
    p = RegulatoryApprovalSearchParams.parse(params)
    search_criteria = DocumentSearchCriteria.parse(p)

    if len(p.terms) < 1:
        logger.error("Terms required for trials search: %s", p.terms)
        return []

    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "approvals",
        }
    )

    async def _search(limit: int):
        where = get_where_clause(search_criteria)
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
