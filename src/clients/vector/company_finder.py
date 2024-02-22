"""
Company client
"""

from datetime import date
import json
import logging
import time
from typing import Sequence
from pydash import flatten, group_by, omit
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

MIN_YEAR = 2000
RECENCY_DECAY_FACTOR = 2


class SemanticCompanyFinder:
    def __init__(self):
        self.gpt_client = GptApiClient()

    @staticmethod
    async def _get_companies_vector(companies: Sequence[str]) -> list[float]:
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

    @staticmethod
    async def _form_vector(
        description: str | None, companies: Sequence[str]
    ) -> list[float]:
        """
        Get the vector for a description & list of companies
        """
        nlp = get_transformer_nlp()

        vectors: list[list[float]] = []

        if description is not None:
            description_doc = nlp(description)
            vectors.append(description_doc.vector.tolist())

        if len(companies) > 0:
            company_vector = await SemanticCompanyFinder._get_companies_vector(
                companies
            )
            vectors.append(company_vector)

        combined_vector: list[float] = (
            torch.stack([torch.tensor(v) for v in vectors]).mean(dim=0).tolist()
        )

        return combined_vector

    @staticmethod
    def _get_fields(
        companies: Sequence[str],
        description: str | None,
        vector: Sequence[float],
        exag_factor: int,
        recency_decay_factor: int = RECENCY_DECAY_FACTOR,
    ) -> list[str]:
        """
        Get fields for the query that differ based on the presence of companies and description
        """
        current_year = date.today().year

        common_fields = [
            "owner.id as id",
            "owner.name as name",
            "(owner.acquisition_count > 0) AS is_acquirer",
            "(owner.acquisition_count = 0 AND owner.owner_type='INDUSTRY') AS is_competition",
            "financials.symbol as symbol",
            "ARRAY_AGG(top_patents.id) AS ids",
            "COUNT(title) AS count",
            "ARRAY_AGG(title) AS titles",
            f"MIN({current_year}-date_part('year', priority_date))::int AS min_age",
            f"ROUND(AVG({current_year}-date_part('year', priority_date))) AS avg_age",
            f"ROUND(POW((1 - (AVG(top_patents.vector) <=> owner.vector)), {exag_factor})::numeric, 2) AS wheelhouse_score",
        ]
        company_fields = [
            f"ROUND(POW((1 - (owner.vector <=> '{vector}')), {exag_factor})::numeric, 2) AS relevance_score",
        ]

        description_fields = [
            f"ROUND(AVG(relevance_score), 2) AS relevance_score",
            f"""
                ROUND(
                    SUM(
                        relevance_score
                        * POW(
                            GREATEST(0.0, ((date_part('year', priority_date) - 2000) / 24)),
                            {recency_decay_factor}
                        )
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

    @staticmethod
    def _form_query(
        p: CompanyFinderParams,
        description: str | None,
        vector: list[float],
    ) -> str:
        fields = SemanticCompanyFinder._get_fields(
            p.similar_companies, description, vector, p.exag_factor
        )
        query = rf"""
            select {', '.join(fields)}
            FROM (
                -- this is probably sub-optimal
                -- groups by title to avoid dups (also hack-y)
                SELECT
                    MAX(id) as id,
                    ARRAY_AGG(id) as ids,
                    AVG(relevance_score) as relevance_score,
                    AVG(vector) as vector,
                    MIN(priority_date) as priority_date,
                    MAX(title) as title
                FROM (
                        SELECT
                            id,
                            family_id,
                            title,
                            POW((1 - (vector <=> '{vector}')), {p.exag_factor})::numeric as relevance_score,
                            vector,
                            priority_date
                        FROM patent
                        ORDER BY (1 - (vector <=> '{vector}')) DESC
                        LIMIT {p.k}
                ) embed
                WHERE relevance_score >= {p.min_relevance_score}
                GROUP BY embed.family_id
            ) top_patents
            JOIN ownable on ownable.patent_id=top_patents.id
            JOIN owner on owner.id=ownable.owner_id
            LEFT JOIN financials ON financials.owner_id=owner.id
            GROUP BY owner.name, owner.id, owner.owner_type, financials.symbol
        """
        return query

    @staticmethod
    def _compute_scores(companies: Sequence[CompanyRecord]) -> tuple[float, float]:
        total = sum([c.score for c in companies])
        exit_score = (
            (sum([c.score for c in companies if c.is_acquirer]) / total)
            if total > 0
            else 0
        )
        competition_score = (
            (sum([c.score for c in companies if c.is_competition]) / total)
            if total > 0
            else 0
        )
        return exit_score, competition_score

    @staticmethod
    async def _fetch_company_reports(
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
            WHERE patent.id = ANY($1)
            AND ownable.owner_id = ANY($2)
            AND ownable.patent_id=patent.id
            AND date_part('year', priority_date) >= {min_year}
            GROUP BY owner_id, date_part('year', priority_date)
        """
        async with prisma_context(300) as db:
            report = await db.query_raw(report_query, patent_ids, owner_ids)

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

    async def __call__(self, p: CompanyFinderParams) -> FindCompanyResult:
        """
        A specific method to find potential buyers for IP / companies

        Does a cosine similarity search on the description and returns the top k
            with scoring based on recency and relevance

        Args:
            p.description: description of the IP
            p.companies: list of companies to search
            p.k: number of nearest neighbors to consider
            p.use_gpt_expansion: whether to use GPT to expand the description (mostly for testing)
            p.min_relevance_score: minimum relevance score to consider
            p.exag_factor: factor to exaggerate cosine similarity
        """
        start = time.monotonic()

        if p.use_gpt_expansion and p.description is not None:
            logger.info("Using GPT to expand description")
            description = await self.gpt_client.generate_ip_description(p.description)
        else:
            description = p.description

        vector = await self._form_vector(description, p.similar_companies)
        query = self._form_query(p, description, vector)

        async with prisma_context(300) as db:
            records = await db.query_raw(query)

        owner_ids = tuple([record["id"] for record in records])
        patent_ids = tuple(flatten([record["ids"] for record in records]))
        report_map = await self._fetch_company_reports(patent_ids, owner_ids)

        companies = sorted(
            [
                CompanyRecord(
                    **omit(record, "score"),
                    activity=[v.count for v in report_map[record["id"]]],
                    count_by_year=report_map[record["id"]],
                    score=(
                        record["score"]
                        if record["score"] is not None
                        else record["relevance_score"]
                    ),
                )
                for record in records
            ],
            key=lambda x: x.score,
            reverse=True,
        )

        exit_score, competition_score = self._compute_scores(companies)

        logger.info(
            "Find took %s seconds (%s)",
            round(time.monotonic() - start, 2),
            len(companies),
        )

        return FindCompanyResult(
            companies=companies,
            description=description or "",
            exit_score=exit_score,
            competition_score=competition_score,
        )
