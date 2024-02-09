"""
Patent client
"""

from datetime import date
import logging
import time

from pydantic import BaseModel

from clients.low_level.prisma import prisma_client
from core.ner.spacy import get_transformer_nlp


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMILARITY_EXAGGERATION_FACTOR = 100
DEFAULT_K = 1000


class BuyerRecord(BaseModel):
    name: str
    ids: list[str]
    count: int
    titles: list[str]
    # terms: list[str]
    min_age: int
    avg_age: float
    max_relevance_score: float
    avg_relevance_score: float
    score: float


async def find_buyers(query: str, k: int = DEFAULT_K) -> list[BuyerRecord]:
    """
    A specific method to find potential buyers for IP
    """
    start = time.monotonic()
    client = await prisma_client(120)
    nlp = get_transformer_nlp()
    query_doc = nlp(query)
    vector = query_doc.vector.tolist()

    current_year = date.today().year

    query = rf"""
            select
                owner.name as name,
                ARRAY_AGG(patent.id) AS ids,
                COUNT(title) AS count,
                ARRAY_AGG(title) AS titles,
                MIN({current_year}-date_part('year', priority_date))::int AS min_age,
                ROUND(avg({current_year}-date_part('year', priority_date))) AS avg_age,
                MAX(1 - (vector <=> '{vector}')) AS max_relevance_score,
                (1 - (AVG(vector) <=> '{vector}')) AS avg_relevance_score,
                SUM(
                    POW((1 - (vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR}) -- cosine similarity
                    * POW(((date_part('year', priority_date) - 2000) / 24), 2) -- recency
                ) AS score
            FROM ownable, owner, patent
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
                        LIMIT {k}
                ) embed
                GROUP BY embed.title
            )
            GROUP BY owner.name
        """

    records = await client.query_raw(query)

    potential_buyers = [BuyerRecord(**record) for record in records]

    logger.info(
        "Find took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(potential_buyers),
    )

    return potential_buyers
