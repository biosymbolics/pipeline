"""
Trials client
"""

import logging
from typing import Sequence
from prisma.types import TrialWhereInput

from clients.documents.utils import (
    get_matching_doc_ids,
    get_search_clause,
)
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import TrialSearchParams
from typings.client import (
    DocumentSearchCriteria,
)
from typings.documents.common import DocType
from typings.documents.trials import ScoredTrial
from utils.string import get_id

from .trials_client import find_many

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_where_clause(
    p: DocumentSearchCriteria,
    term_matching_ids: Sequence[str] | None = None,
    description_ids: Sequence[str] | None = None,
) -> TrialWhereInput:
    is_id_search = any([t.startswith("NCT") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("NCT") for t in p.terms]):
        raise ValueError("ID search; all terms must be NCT.*")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    return get_search_clause(
        DocType.trial,
        p,
        term_matching_ids,
        description_ids,
        return_type=TrialWhereInput,
    )


async def search(params: TrialSearchParams) -> list[ScoredTrial]:
    """
    Search trials by terms
    """
    p = TrialSearchParams.parse(params)
    search_criteria = DocumentSearchCriteria.parse(p)

    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "trials",
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
        trials = await _search(limit=p.limit)
        return trials

    return await retrieve_with_cache_check(
        _search,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredTrial(**t) for t in storage_decoder(str_data)],
    )
