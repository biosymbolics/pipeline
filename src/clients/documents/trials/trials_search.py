"""
Trials client
"""

from datetime import datetime
import logging
from prisma.types import TrialWhereInput, TrialWhereInputRecursive1

from clients.documents.utils import get_term_clause
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import TrialSearchParams
from typings.client import (
    DocumentSearchCriteria,
    DocumentSearchParams,
)
from typings.documents.trials import ScoredTrial
from utils.string import get_id

from .trials_client import find_many

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_where_clause(p: DocumentSearchCriteria) -> TrialWhereInput:
    is_id_search = any([t.startswith("NCT") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("NCT") for t in p.terms]):
        raise ValueError("ID search; all terms must be NCT.*")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    term_clause = get_term_clause(p, TrialWhereInputRecursive1)

    where: TrialWhereInput = {
        "AND": [
            term_clause,
            {
                "start_date": {
                    "gte": datetime(p.start_year, 1, 1),
                    "lte": datetime(p.end_year, 1, 1),
                }
            },
        ],
    }

    return where


async def search(params: DocumentSearchParams | TrialSearchParams) -> list[ScoredTrial]:
    """
    Search trials by terms

    Args:
        p.terms (Sequence[str]): list of terms to search for
        p.include (TrialInclude, optional): whether to include assignees, inventors, interventions, indications. Defaults to DEFAULT_PATENT_INCLUDE.
        p.start_year (int, optional): minimum priority date year. Defaults to DEFAULT_START_YEAR.
        p.end_year (int, optional): maximum priority date year. Defaults to DEFAULT_END_YEAR.
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.term_fields (Sequence[TermField], optional): which fields to search for terms in. Defaults to DEFAULT_TERM_FIELDS.
        p.limit (int, optional): max results to return.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.
    """
    p = TrialSearchParams.parse(params)
    search_criteria = DocumentSearchCriteria.parse(p)

    if len(p.terms) < 1:
        logger.error("Terms required for trials search: %s", p.terms)
        return []

    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "trials",
        }
    )

    async def _search(limit: int):
        where = get_where_clause(search_criteria)
        return await find_many(where=where, include=p.include, take=limit)

    if p.skip_cache == True:
        trials = await _search(limit=p.limit)
        return trials

    return await retrieve_with_cache_check(
        _search,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredTrial(**t) for t in storage_decoder(str_data)],
    )
