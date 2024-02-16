"""
Company client
"""

from datetime import date
import json
import logging
import time
from typing import Sequence
from pydash import flatten, group_by
import torch

from clients.low_level.prisma import prisma_context
from clients.openai.gpt_client import GptApiClient
from constants.documents import MAX_DATA_YEAR
from core.ner.spacy import get_transformer_nlp
from typings.client import CompanyFinderParams
from .types import (
    CompanyRecord,
    FindCompanyResult,
    CountByYear,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SIMILARITY_EXAGGERATION_FACTOR = 50
MIN_YEAR = 2000


async def fetch_company_reports(
    patent_ids: Sequence[str], owner_ids: Sequence[str], min_year: int = MIN_YEAR
) -> dict[str, list[CountByYear]]:
    """
    Fetches company reports for a set of patent and owner ids
    """
    report_query = f"""
        SELECT
            owner_id as id,
            date_part('year', priority_date) as year,
            count(*) as count
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
        k: {v["year"]: CountByYear(**v) for v in vals}
        for k, vals in group_by(report, "id").items()
    }
    return {
        k: [
            vals.get(y) or CountByYear(year=y, count=0)
            for y in range(min_year, MAX_DATA_YEAR)
        ]
        for k, vals in result_map.items()
    }


async def get_companies_vector(companies: Sequence[str]) -> list[float]:
    """
    Get the avg vector for a list of companies
    """
    query = f"""
        SELECT AVG(vector)::text vector FROM owner
        WHERE name=ANY($1)
    """
    async with prisma_context(300) as db:
        result = await db.query_raw(query, companies)

    return json.loads(result[0]["vector"])


async def get_vector(description: str | None, companies: Sequence[str]) -> list[float]:
    """
    Get the vector for a description & list of companies
    """
    nlp = get_transformer_nlp()

    vectors: list[list[float]] = []

    if description is not None:
        description_doc = nlp(description)
        vectors.append(description_doc.vector.tolist())

    if len(companies) > 0:
        company_vector = await get_companies_vector(companies)
        vectors.append(company_vector)

    combined_vector: list[float] = (
        torch.stack([torch.tensor(v) for v in vectors]).mean(dim=0).tolist()
    )

    return combined_vector


def get_fields(
    companies: Sequence[str], description: str | None, vector: Sequence[float]
) -> list[str]:
    """
    Get fields for the query that differ based on the presence of companies and description
    """
    current_year = date.today().year

    common_fields = [
        "owner.id as id",
        "owner.name as name",
        "COALESCE(financials.symbol, '') as symbol",
        "ARRAY_AGG(patent.id) AS ids",
        "COUNT(title) AS count",
        "ARRAY_AGG(title) AS titles",
        f"MIN({current_year}-date_part('year', priority_date))::int AS min_age",
        f"ROUND(AVG({current_year}-date_part('year', priority_date))) AS avg_age",
        f"ROUND(POW((1 - (AVG(patent.vector) <=> owner.vector)), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 2) AS wheelhouse_score",
    ]
    company_fields = [
        f"ROUND(POW((1 - (owner.vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 2) AS relevance_score",
        f"ROUND(POW((1 - (owner.vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 2) AS score",
    ]

    description_fields = [
        f"ROUND(POW((1 - (AVG(patent.vector) <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR})::numeric, 2) AS relevance_score",
        f"""
            ROUND(
                SUM(
                    POW((1 - (patent.vector <=> '{vector}')), {SIMILARITY_EXAGGERATION_FACTOR}) -- cosine similarity
                    * POW(((date_part('year', priority_date) - 2000) / 24), 2) -- recency
                )::numeric, 2
            ) AS score
        """,
    ]

    if description:
        # wins over companies
        return common_fields + description_fields

    if companies:
        return common_fields + company_fields

    raise ValueError("No companies or description")


async def find_companies(p: CompanyFinderParams) -> FindCompanyResult:
    """
    A specific method to find potential buyers for IP / companies

    Does a cosine similarity search on the description and returns the top k
        with scoring based on recency and relevance

    Args:
        description: description of the IP
        companies: list of companies to search
        k: number of nearest neighbors to consider
        use_gpt_expansion: whether to use GPT to expand the description (mostly for testing)
    """
    start = time.monotonic()

    if p.use_gpt_expansion and p.description is not None:
        logger.info("Using GPT to expand description")
        gpt_client = GptApiClient()
        description = await gpt_client.generate_ip_description(p.description)
    else:
        description = p.description

    vector = await get_vector(description, p.companies)
    fields = get_fields(p.companies, description, vector)

    query = rf"""
        select {', '.join(fields)}
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
                    LIMIT {p.k}
            ) embed
            GROUP BY embed.title
        )
        GROUP BY owner.name, owner.id, financials.symbol
    """

    async with prisma_context(300) as db:
        records = await db.query_raw(query)

    owner_ids = tuple([record["id"] for record in records])
    patent_ids = tuple(flatten([record["ids"] for record in records]))
    report_map = await fetch_company_reports(patent_ids, owner_ids)

    companies = sorted(
        [
            CompanyRecord(
                **record,
                activity=[v.count for v in report_map[record["id"]]],
                count_by_year=report_map[record["id"]],
            )
            for record in records
        ],
        key=lambda x: x.score,
        reverse=True,
    )

    logger.info(
        "Find took %s seconds (%s)",
        round(time.monotonic() - start, 2),
        len(companies),
    )

    return FindCompanyResult(companies=companies, description=description or "")
