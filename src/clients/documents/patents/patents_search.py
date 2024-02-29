"""
Patent client
"""

from datetime import datetime
import logging
from typing import Sequence
from prisma.types import (
    PatentWhereInput,
    PatentWhereInputRecursive1,
)

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from constants.core import DEFAULT_VECTORIZATION_MODEL
from typings.documents.patents import ScoredPatent
from typings.client import (
    DocumentSearchCriteria,
    DocumentSearchParams,
    PatentSearchParams,
)
from utils.string import get_id

from .patents_client import find_many

from ..utils import get_term_clause


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_where_clause(
    p: DocumentSearchCriteria, description_ids: Sequence[str] | None = None
) -> PatentWhereInput:
    is_id_search = any([t.startswith("WO-") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in p.terms]):
        raise ValueError("ID search; all terms must be WO-.*")

    if description_ids is not None and is_id_search:
        raise ValueError("Cannot search by description and id")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    term_clause = get_term_clause(p, PatentWhereInputRecursive1)

    where: PatentWhereInput = {
        "AND": [
            term_clause,
            {
                "priority_date": {
                    "gte": datetime(p.start_year, 1, 1),
                    "lte": datetime(p.end_year, 1, 1),
                }
            },
            {"id": {"in": list(description_ids)}} if description_ids else {},
        ],
    }

    return where


async def get_description_ids(description: str, k: int) -> list[str]:
    """
    Get patent ids within K nearest neighbors of a vectorized description

    Args:
        description (str): a description of the desired patents
        k (int): k nearest neighbors
    """
    # lazy import
    from core.ner.spacy import get_transformer_nlp

    logger.info("Searching patents by description (slow-ish)")
    nlp = get_transformer_nlp(DEFAULT_VECTORIZATION_MODEL)
    vector = nlp(description).vector.tolist()

    query = f"""
        SELECT id FROM patent
        ORDER BY (1 - (vector <=> '{vector}')) DESC
        LIMIT {k}
    """

    async with prisma_context(300) as db:
        results = await db.query_raw(query)
    return [r["id"] for r in results]


async def search(
    params: DocumentSearchParams | PatentSearchParams,
) -> list[ScoredPatent]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        p.terms (Sequence[str]): list of terms to search for
        p.description (Sequence[str], optional): a description of the desired patents. Defaults to None.
        p.k (int, optional): k nearest neighbors. used only with `description`. Defaults to DEFAULT_K.
        p.include (PatentInclude, optional): whether to include assignees, inventors, interventions, indications. Defaults to DEFAULT_PATENT_INCLUDE.
        p.start_year (int, optional): minimum priority date year. Defaults to DEFAULT_START_YEAR.
        p.end_year (int, optional): maximum priority date year. Defaults to DEFAULT_END_YEAR.
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.term_fields (Sequence[TermField], optional): which fields to search for terms in. Defaults to DEFAULT_TERM_FIELDS.
        p.limit (int, optional): max results to return.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.

    Returns: a list of matching patent applications
    """
    p = PatentSearchParams.parse(params)

    search_criteria = DocumentSearchCriteria.parse(p)
    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "patents",
        }
    )

    # if a description is provided, get the ids of the nearest neighbors
    if p.description:
        description_ids = await get_description_ids(p.description, k=p.k)
    else:
        description_ids = None

    where = get_where_clause(search_criteria, description_ids)

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
