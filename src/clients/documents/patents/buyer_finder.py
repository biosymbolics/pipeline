"""
Buyer finder client
"""

from datetime import date
import logging
import time

from pydash import flatten, group_by
from clients.documents.patents.constants import DEFAULT_BUYER_K


from clients.low_level.prisma import prisma_client
from clients.openai.gpt_client import GptApiClient
from core.ner.spacy import get_transformer_nlp
from typings.core import ResultBase


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMILARITY_EXAGGERATION_FACTOR = 50


class RelevanceByYear(ResultBase):
    year: int
    relevance: float


class BuyerRecord(ResultBase):
    id: int
    name: str
    ids: list[str]
    count: int
    symbol: str | None
    titles: list[str]
    # terms: list[str]
    min_age: int
    avg_age: float
    activity: list[float]
    max_relevance_score: float
    avg_relevance_score: float
    relevance_by_year: list[RelevanceByYear]
    score: float


class FindBuyerResult(ResultBase):
    buyers: list[BuyerRecord]
    description: str  # included since it could be a result of expansion


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
    client = await prisma_client(120)
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

    records = await client.query_raw(query)

    owner_ids = tuple([record["id"] for record in records])
    patent_ids = tuple(flatten([record["ids"] for record in records]))

    report_query = f"""
        SELECT
            owner_id as id,
            date_part('year', priority_date) as year,
            count(*) as relevance
        FROM patent, ownable
        WHERE patent.id in {patent_ids}
        AND ownable.owner_id in {owner_ids}
        AND ownable.patent_id=patent.id
        GROUP BY owner_id, date_part('year', priority_date)
    """
    report = await client.query_raw(report_query)
    result_map = {
        k: {v["year"]: RelevanceByYear(**v) for v in vals}
        for k, vals in group_by(report, "id").items()
    }
    report_map = {
        k: [
            vals.get(y) or RelevanceByYear(year=y, relevance=0.0)
            for y in range(2000, 2024)
        ]
        for k, vals in result_map.items()
    }

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
