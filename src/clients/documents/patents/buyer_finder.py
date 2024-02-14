"""
Buyer finder client
"""

from datetime import date
import logging
import time
from typing import Sequence
from pydash import flatten, group_by

from clients.low_level.prisma import prisma_context
from clients.openai.gpt_client import GptApiClient
from constants.documents import MAX_DATA_YEAR
from constants.patents import DEFAULT_BUYER_K
from core.ner.spacy import get_transformer_nlp
from .types import (
    BuyerRecord,
    FindBuyerResult,
    RelevanceByYear,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMILARITY_EXAGGERATION_FACTOR = 50
MIN_YEAR = 2000


async def fetch_buyer_reports(
    patent_ids: Sequence[str], owner_ids: Sequence[str], min_year: int = MIN_YEAR
) -> dict[str, list[RelevanceByYear]]:
    """
    Fetches buyer reports for a set of patent and owner ids
    """
    report_query = f"""
        SELECT
            owner_id as id,
            date_part('year', priority_date) as year,
            count(*) as relevance
        FROM patent, ownable
        WHERE patent.id in {patent_ids}
        AND ownable.owner_id in {owner_ids}
        AND ownable.patent_id=patent.id
        AND date_part('year', priority_date) >= {min_year}
        GROUP BY owner_id, date_part('year', priority_date)
    """

    async with prisma_context(300) as db:
        report = await db.query_raw(report_query)

    result_map = {
        k: {v["year"]: RelevanceByYear(**v) for v in vals}
        for k, vals in group_by(report, "id").items()
    }
    return {
        k: [
            vals.get(y) or RelevanceByYear(year=y, relevance=0.0)
            for y in range(min_year, MAX_DATA_YEAR)
        ]
        for k, vals in result_map.items()
    }


async def find_buyers(
    description: str, knn: int = DEFAULT_BUYER_K, use_gpt_expansion: bool = False
) -> FindBuyerResult:
    """
    A specific method to find potential buyers for IP

    Does a cosine similarity search on the description and returns the top k
        with scoring based on recency and relevance

    Args:
        description: description of the IP
        k: number of nearest neighbors to consider
        use_gpt_expansion: whether to use GPT to expand the description (mostly for testing)
    """
    start = time.monotonic()
    nlp = get_transformer_nlp()

    if use_gpt_expansion:
        logger.info("Using GPT to expand description")
        gpt_client = GptApiClient()
        description = await gpt_client.generate_ip_description(description)

    description_doc = nlp(description)
    vector = description_doc.vector.tolist()

    current_year = date.today().year

    query = rf"""
        select
            owner.id as id,
            owner.name as name,
            COALESCE(financials.symbol, '') as symbol,
            ARRAY_AGG(patent.id) AS ids,
            COUNT(title) AS count,
            ARRAY_AGG(title) AS titles,
            MIN({current_year}-date_part('year', priority_date))::int AS min_age,
            ROUND(AVG({current_year}-date_part('year', priority_date))) AS avg_age,
            ROUND(POW(MAX(1 - (vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 3) AS max_relevance_score,
            ROUND(POW((1 - (AVG(vector) <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 3) AS avg_relevance_score,
            ROUND(
                SUM(
                    POW((1 - (vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR}) -- cosine similarity
                    * POW(((date_part('year', priority_date) - 2000) / 24), 2) -- recency
                )::numeric, 2
            ) AS score
        FROM ownable, patent, owner
        LEFT JOIN financials ON financials.owner_id=owner.id
        WHERE ownable.patent_id=patent.id
        AND owner.id=ownable.owner_id
        AND owner.owner_type in ('INDUSTRY', 'INDUSTRY_LARGE')
        AND patent.id IN (
            -- this is probably sub-optimal
            -- groups by title to avoid dups (also hack-y)
            SELECT MAX(id)
            FROM (
                    SELECT id, title
                    FROM patent
                    ORDER BY (1 - (vector <=> '{vector}')) DESC
                    LIMIT {knn}
            ) embed
            GROUP BY embed.title
        )
        GROUP BY owner.name, owner.id, financials.symbol
    """

    async with prisma_context(300) as db:
        records = await db.query_raw(query)

    owner_ids = tuple([record["id"] for record in records])
    patent_ids = tuple(flatten([record["ids"] for record in records]))
    report_map = await fetch_buyer_reports(patent_ids, owner_ids)

    potential_buyers = sorted(
        [
            BuyerRecord(
                **record,
                activity=[v.relevance for v in report_map[record["id"]]],
                relevance_by_year=report_map[record["id"]],
            )
            for record in records
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    logger.info(
        "Find took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(potential_buyers),
    )

    return FindBuyerResult(buyers=potential_buyers, description=description)
