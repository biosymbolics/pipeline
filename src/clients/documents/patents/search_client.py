"""
Patent client
"""
from functools import partial
import logging
from typing import Sequence
from prisma.client import Prisma
from prisma.types import (
    PatentWhereInput,
    PatentWhereInputRecursive1,
)

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from typings.documents.patents import ScoredPatent
from typings.client import (
    DocumentSearchCriteria,
    DocumentSearchParams,
    PatentSearchParams,
    QueryType,
)
from utils.string import get_id

from .client import find_many
from .utils import get_max_priority_date

from ..utils import get_where_clause as get_term_clause


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXEMPLAR_SIMILARITY_THRESHOLD = 0.7


async def get_exemplar_embeddings(exemplar_patents: Sequence[str]) -> list[str]:
    """
    Get embeddings for exemplar patents
    """
    async with prisma_context(300) as client:
        results = await Prisma.query_raw(
            client,
            "SELECT embeddings FROM patent WHERE id = ANY($1)",
            exemplar_patents,
        )

    # exemplar_embeddings = (
    #     await get_exemplar_embeddings(exemplar_patents)
    #     if len(exemplar_patents) > 0
    #     else []
    # )
    # exemplar_criterion = [
    #     f"(1 - (embeddings <=> '{e}')) > {EXEMPLAR_SIMILARITY_THRESHOLD}"
    #     for e in exemplar_embeddings
    # ]
    # wheres.append(f"AND ({f' {query_type} '.join(exemplar_criterion)})")
    # cosine_scores = [f"(1 - (embeddings <=> '{e}'))" for e in exemplar_embeddings]
    # froms.append(f", unnest (ARRAY[{','.join(cosine_scores)}]) cosine_scores")
    # fields.append("AVG(cosine_scores) as exemplar_similarity")
    return [r["embeddings"] for r in results]


def get_where_clause(p: DocumentSearchCriteria) -> PatentWhereInput:
    is_id_search = all([t.startswith("WO-") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in p.terms]):
        raise ValueError("ID search must be all WO-")

    if is_id_search:
        return {"id": {"in": list(p.terms)}}

    term_clause = get_term_clause(p, PatentWhereInputRecursive1)

    where: PatentWhereInput = {
        "AND": [
            term_clause,
            {"priority_date": {"gte": get_max_priority_date(0)}},  # TODO
        ],
    }

    return where


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
        p.exemplar_patents (Sequence[str], optional): list of exemplar patents to search for. Defaults to [].
        p.include (PatentInclude, optional): whether to include assignees, inventors, interventions, indications. Defaults to DEFAULT_PATENT_INCLUDE.
        p.start_year (int, optional): minimum priority date year. Defaults to DEFAULT_START_YEAR.
        p.end_year (int, optional): maximum priority date year. Defaults to DEFAULT_END_YEAR.
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.term_fields (Sequence[TermField], optional): which fields to search for terms in. Defaults to DEFAULT_TERM_FIELDS.
        p.limit (int, optional): max results to return.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.

    Returns: a list of matching patent applications

    Example:
    ```
    from clients.documents.patents import search_client
    from typings.client import PatentSearchParams
    p = search_client.search(PatentSearchParams(terms=['migraine disorders'], skip_cache=True, limit=5))
    [t.search_rank for t in p]
    ```
    """
    p = PatentSearchParams.parse(params)
    search_criteria = DocumentSearchCriteria.parse(p)
    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "patents",
        }
    )

    async def _search(limit: int):
        where = get_where_clause(search_criteria)

        return await find_many(
            where=where,
            include=p.include,
            take=limit,
        )

    if p.skip_cache == True:
        patents = await _search(limit=p.limit)
        return patents

    return await retrieve_with_cache_check(
        _search,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredPatent(**p) for p in storage_decoder(str_data)],
    )
