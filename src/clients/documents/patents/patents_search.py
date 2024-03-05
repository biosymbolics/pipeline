"""
Patent client
"""

import logging
from typing import Sequence
from prisma.types import PatentWhereInput

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from typings.documents.common import DocType
from typings.documents.patents import ScoredPatent
from typings.client import (
    DocumentSearchCriteria,
    PatentSearchParams,
)
from utils.string import get_id

from .patents_client import find_many

from ..utils import (
    get_matching_doc_ids,
    get_search_clause,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_where_clause(
    p: DocumentSearchCriteria,
    term_match_ids: Sequence[str] | None = None,
    vector_match_ids: Sequence[str] | None = None,
) -> PatentWhereInput:
    is_id_search = any([t.startswith("WO-") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in p.terms]):
        raise ValueError("ID search; all terms must be WO-.*")

    if vector_match_ids is not None and is_id_search:
        raise ValueError("Cannot search by description and id")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    return get_search_clause(
        DocType.patent,
        p,
        term_match_ids,
        vector_match_ids,
        return_type=PatentWhereInput,
    )


async def search(
    params: PatentSearchParams,
) -> list[ScoredPatent]:
    """
    Search patents by terms
    """
    p = PatentSearchParams.parse(params)

    search_criteria = DocumentSearchCriteria.parse(p)
    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "patents",
        }
    )

    term_match_ids, vector_match_ids = await get_matching_doc_ids(
        p,
        [DocType.patent],
    )

    where = get_where_clause(search_criteria, term_match_ids, vector_match_ids)

    async def _search(limit: int):
        return await find_many(
            where=where,
            include=p.include,
            # big perf improvement over default sort (id)
            order={"priority_date": "desc"},
            take=limit,
        )

    if p.skip_cache == True:
        logger.info("Skipping cache for %s", key)
        patents = await _search(limit=p.limit)
        return patents

    return await retrieve_with_cache_check(
        _search,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredPatent(**p) for p in storage_decoder(str_data)],
    )
