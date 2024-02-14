"""
Patent client
"""

from datetime import datetime
import logging
from prisma.types import (
    PatentWhereInput,
    PatentWhereInputRecursive1,
)

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
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

EXEMPLAR_SIMILARITY_THRESHOLD = 0.7


def get_where_clause(p: DocumentSearchCriteria) -> PatentWhereInput:
    is_id_search = any([t.startswith("WO-") for t in p.terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in p.terms]):
        raise ValueError("ID search; all terms must be WO-.*")

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
