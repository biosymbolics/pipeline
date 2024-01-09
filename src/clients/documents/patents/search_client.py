"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence
from prisma.client import Prisma
from prisma.types import PatentWhereInput, PatentWhereInputRecursive2

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import get_prisma_client
from typings.patents import ScoredPatent as PatentApplication
from typings.client import PatentSearchParams, QueryType, TermField
from utils.string import get_id

from .client import find_many
from .utils import get_max_priority_date


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50
EXEMPLAR_SIMILARITY_THRESHOLD = 0.7


async def get_exemplar_embeddings(exemplar_patents: Sequence[str]) -> list[str]:
    """
    Get embeddings for exemplar patents
    """
    async with get_prisma_client(300) as client:
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


def get_term_clause(
    terms: Sequence[str], query_type: QueryType
) -> list[PatentWhereInputRecursive2]:
    if query_type == "AND":
        return [
            {
                "AND": [
                    {"interventions": {"some": {"canonical_name": {"in": terms}}}},
                    {"indications": {"some": {"canonical_name": {"in": terms}}}},
                ],
                "AND": [
                    {"interventions": {"some": {"canonical_name": {"in": terms}}}},
                    {"assignees": {"some": {"canonical_name": {"in": terms}}}},
                ],
                "AND": [
                    {"indications": {"some": {"canonical_name": {"in": terms}}}},
                    {"assignees": {"some": {"canonical_name": {"in": terms}}}},
                ],
            }
        ]

    return [
        {"interventions": {"some": {"canonical_name": {"in": terms}}}},
        {"indications": {"some": {"canonical_name": {"in": terms}}}},
    ]


def get_where_clause(
    terms: Sequence[str], query_type: QueryType, min_patent_years: int
) -> PatentWhereInput:
    is_id_search = all([t.startswith("WO-") for t in terms])

    # require homogeneous search
    if is_id_search and any([not t.startswith("WO-") for t in terms]):
        raise ValueError("ID search must be all WO-")

    if is_id_search:
        return {"id": {"in": list(terms)}}

    lower_terms = [t.lower() for t in terms]
    term_clause = get_term_clause(lower_terms, query_type)

    where: PatentWhereInput = {
        "AND": [
            {"OR": term_clause},
            {"priority_date": {"gte": get_max_priority_date(int(min_patent_years))}},
        ],
    }

    return where


async def _search(
    terms: Sequence[str],
    exemplar_patents: Sequence[str] = [],
    query_type: QueryType = "OR",
    min_patent_years: int = 10,
    term_field: TermField = "terms",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[PatentApplication]:
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

    where = get_where_clause(terms, query_type, min_patent_years)

    patents = await find_many(
        where=where,
        include={
            "assignees": True,
            "inventors": True,
            "interventions": True,
            "indications": True,
        },
        take=limit,
    )

    return patents


async def search(p: PatentSearchParams) -> list[PatentApplication]:
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
        p.term_field (TermField, optional): which field to search on. Defaults to "terms".
                Other values are `instance_rollup` (which are rollup terms at a high level of specificity, e.g. "Aspirin 50mg" might have a rollup term of "Aspirin")
                and `category_rollup` (wherein "Aspirin 50mg" might have a rollup category of "NSAIDs")
        p.limit (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.

    Returns: a list of matching patent applications

    Example:
    ```
    from clients.patents import search_client
    from handlers.patents.types import PatentSearchParams
    p = search_client.search(PatentSearchParams(terms=['migraine disorders'], skip_cache=True, limit=5))
    [t.search_rank for t in p]
    ```
    """
    args = {
        "terms": p.terms,
        "exemplar_patents": p.exemplar_patents,
        "query_type": p.query_type,
        "min_patent_years": p.min_patent_years,
        "term_field": p.term_field,
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

    return retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [
            PatentApplication(**p) for p in storage_decoder(str_data)
        ],
    )
