"""
Trials client
"""

from datetime import datetime
import logging
from typing import Sequence
from prisma.types import TrialWhereInput, TrialWhereInputRecursive1

from clients.documents.utils import (
    get_doc_ids_for_description,
    get_search_clause,
    get_term_clause,
)
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings import TrialSearchParams
from typings.client import (
    DocumentSearchCriteria,
    DocumentSearchParams,
)
from typings.documents.common import DocType
from typings.documents.trials import ScoredTrial
from utils.string import get_id

from .trials_client import find_many

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_where_clause(
    p: DocumentSearchCriteria, description_ids: Sequence[str] | None = None
) -> TrialWhereInput:
    is_id_search = any([t.startswith("NCT") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("NCT") for t in p.terms]):
        raise ValueError("ID search; all terms must be NCT.*")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    return get_search_clause(
        DocType.trial, p, description_ids, return_type=TrialWhereInput
    )


async def search(params: TrialSearchParams) -> list[ScoredTrial]:
    """
    Search trials by terms
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
        # if a description is provided, get the ids of the nearest neighbors
        if p.description:
            vector_matching_ids = await get_doc_ids_for_description(
                p.description, DocType.trial, p.vector_search_params
            )
        else:
            vector_matching_ids = None

        where = get_where_clause(search_criteria, vector_matching_ids)
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
