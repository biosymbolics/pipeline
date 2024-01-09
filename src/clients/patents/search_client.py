"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence
from prisma.client import Prisma
from prisma.models import Patent

from clients.companies import get_financial_map
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import get_prisma_client
from typings.patents import ScoredPatent as PatentApplication
from typings.client import PatentSearchParams, QueryType, TermField
from utils.sql import get_term_sql_query
from utils.string import get_id

from .enrich import enrich_search_result
from .types import QueryPieces
from .utils import get_max_priority_date


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50
EXEMPLAR_SIMILARITY_THRESHOLD = 0.7


def _get_query_pieces(
    terms: Sequence[str],
    exemplar_embeddings: Sequence[str],
    query_type: QueryType,
    min_patent_years: int,
) -> QueryPieces:
    """
    Helper to generate pieces of patent search query
    """
    is_id_search = any([t.startswith("WO-") for t in terms])

    # if ids, ignore most of the standard criteria
    if is_id_search:
        if (
            not all([t.startswith("WO-") for t in terms])
            or len(exemplar_embeddings) > 0
        ):
            raise ValueError("Cannot mix id and (term or exemplar patent) search")

        return QueryPieces(
            fields=["*", "1 as search_rank", "0 as exemplar_similarity"],
            froms=[],
            wheres=[f"patents.id = ANY($1)"],
            params=[terms],
        )

    lower_terms = [t.lower() for t in terms]
    max_priority_date = get_max_priority_date(int(min_patent_years))

    # exp decay scaling for search terms; higher is better
    fields = [
        "*",
        f"ts_rank_cd(search, to_tsquery($1)) AS search_rank",
    ]
    froms = []
    wheres = [
        f"priority_date > '{max_priority_date}'::date",
        "search @@ to_tsquery($1)",
    ]

    if len(exemplar_embeddings) > 0:
        exemplar_criterion = [
            f"(1 - (embeddings <=> '{e}')) > {EXEMPLAR_SIMILARITY_THRESHOLD}"
            for e in exemplar_embeddings
        ]
        wheres.append(f"AND ({f' {query_type} '.join(exemplar_criterion)})")
        cosine_scores = [f"(1 - (embeddings <=> '{e}'))" for e in exemplar_embeddings]
        froms.append(f", unnest (ARRAY[{','.join(cosine_scores)}]) cosine_scores")
        fields.append("AVG(cosine_scores) as exemplar_similarity")
    else:
        fields.append("0 as exemplar_similarity")

    return QueryPieces(
        fields=fields,
        froms=froms,
        wheres=wheres,
        params=[get_term_sql_query(terms, query_type)],
    )


async def get_exemplar_embeddings(exemplar_patents: Sequence[str]) -> list[str]:
    """
    Get embeddings for exemplar patents
    """
    async with get_prisma_client(300) as client:
        results = await Prisma.query_raw(
            client,
            "SELECT embeddings FROM patents WHERE id = ANY($1)",
            exemplar_patents,
        )
    return [r["embeddings"] for r in results]


async def _search(
    terms: Sequence[str],
    exemplar_patents: Sequence[str] = [],
    query_type: QueryType = "AND",
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
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    exemplar_embeddings = (
        await get_exemplar_embeddings(exemplar_patents)
        if len(exemplar_patents) > 0
        else []
    )
    qp = _get_query_pieces(terms, exemplar_embeddings, query_type, min_patent_years)

    query = f"""
        SELECT {", ".join(qp["fields"])}
        FROM patent {", ".join(qp["froms"])}
        WHERE {" AND ".join(qp["wheres"])}
        ORDER BY priority_date desc
        LIMIT {limit}
    """
    print(query)

    async with get_prisma_client(300):
        patents = await Patent.prisma().query_raw(query, *qp["params"])

    ids = [p.id for p in patents]
    financial_map = await get_financial_map(ids, "assignee_patent_id")
    enriched_results = enrich_search_result(patents, financial_map)

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(patents)
    )

    return enriched_results


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
