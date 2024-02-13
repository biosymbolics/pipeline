"""
Buyer finder client
"""

from datetime import date
import logging
import time


from clients.low_level.prisma import prisma_client
from clients.openai.gpt_client import GptApiClient
from core.ner.spacy import get_transformer_nlp
from typings.core import ResultBase


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMILARITY_EXAGGERATION_FACTOR = 100
DEFAULT_K = 1000


class BuyerRecord(ResultBase):
    id: int
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


class FindBuyerResult(ResultBase):
    buyers: list[BuyerRecord]
    description: str  # included since it could be a result of expansion


async def find_buyers(
    description: str, k: int = DEFAULT_K, use_gpt_expansion: bool = False
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
            GROUP BY owner.name, owner.id
        """

    records = await client.query_raw(query)

    logger.info("BUYER RECS %s", records[0:5])
    potential_buyers = sorted(
        [BuyerRecord(**record) for record in records],
        key=lambda x: x.score,
        reverse=True,
    )

    logger.info(
        "Find took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(potential_buyers),
    )

    return FindBuyerResult(buyers=potential_buyers, description=description)
