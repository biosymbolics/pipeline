"""
Patent client
"""
from functools import partial
import logging
from typing import Sequence
from prisma.client import Prisma
from prisma.types import (
    PatentInclude,
    PatentWhereInput,
    PatentWhereInputRecursive1,
)

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from typings import TermField
from typings.documents.patents import ScoredPatent
from typings.client import (
    DEFAULT_PATENT_INCLUDE,
    DEFAULT_QUERY_TYPE,
    DEFAULT_TERM_FIELDS,
    CommonSearchParams,
    PatentSearchParams,
    QueryType,
)
from utils.string import get_id

from .client import find_many
from .utils import get_max_priority_date

from ..utils import get_where_clause as get_term_clause


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
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


def get_where_clause(
    terms: Sequence[str],
    term_fields: Sequence[TermField],
    query_type: QueryType,
) -> PatentWhereInput:
    is_id_search = all([t.startswith("WO-") for t in terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in terms]):
        raise ValueError("ID search must be all WO-")

    if is_id_search:
        return {"id": {"in": list(terms)}}

    term_clause = get_term_clause(
        terms, term_fields, query_type, PatentWhereInputRecursive1
    )

    where: PatentWhereInput = {
        "AND": [
            term_clause,
            {"priority_date": {"gte": get_max_priority_date(0)}},  # TODO
        ],
    }

    return where


async def _search(
    terms: Sequence[str],
    exemplar_patents: Sequence[str] = [],
    query_type: QueryType = DEFAULT_QUERY_TYPE,
    term_fields: Sequence[TermField] = DEFAULT_TERM_FIELDS,
    include: PatentInclude = DEFAULT_PATENT_INCLUDE,
    limit: int = MAX_SEARCH_RESULTS,
) -> list[ScoredPatent]:
    """
    Search patents by terms

    REPL:
    ```
    import asyncio
    from clients.patents.search_client import _search
    with asyncio.Runner() as runner:
        runner.run(_search(["asthma"]))
    ```
    """
    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    where = get_where_clause(terms, term_fields, query_type)

    patents = await find_many(
        where=where,
        include=include,
        take=limit,
    )

    return patents


async def search(params: CommonSearchParams | PatentSearchParams) -> list[ScoredPatent]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        p.terms (Sequence[str]): list of terms to search for
        p.exemplar_patents (Sequence[str], optional): list of exemplar patents to search for. Defaults to [].
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.min_patent_years (int, optional): minimum patent age in years. Defaults to 10.
        p.limit (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
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
    p = PatentSearchParams(**params.__dict__)
    args = {
        "exemplar_patents": p.exemplar_patents,
        "include": p.include,
        "query_type": p.query_type,
        "terms": p.terms,
    }
    key = get_id(
        {
            **args,
            "api": "patents",
        }
    )
    search_partial = partial(_search, **args)

    if p.skip_cache == True:
        patents = await search_partial(limit=p.limit)
        return patents

    return await retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [ScoredPatent(**p) for p in storage_decoder(str_data)],
    )
