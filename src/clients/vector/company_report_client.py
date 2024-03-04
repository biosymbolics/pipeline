"""
Semantic company client
"""

from datetime import date
import logging
import time
from typing import Sequence
from pydash import flatten, group_by, omit

from clients.low_level.prisma import prisma_context
from clients.vector.vector_report_client import VectorReportClient
from constants.documents import MAX_DATA_YEAR
from typings import DocType
from typings.client import CompanyFinderParams, VectorSearchParams
from typings.documents.common import DOC_TYPE_DATE_MAP
from .types import (
    CompanyRecord,
    FindCompanyResult,
    CountByYear,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CompanyReportClient(VectorReportClient):
    async def _fetch_company_reports(
        self, document_ids: Sequence[str], owner_ids: Sequence[int], start_year: int
    ) -> dict[str, dict[int, list[CountByYear]]]:
        """
        Fetches company reports for a set of document and owner ids

        Returns map between owner_id and list of corresponding list[CountByYear]
        """

        def get_report_query(doc_type: DocType):
            date_field = DOC_TYPE_DATE_MAP[doc_type]
            return f"""
                SELECT
                    owner_id AS id,
                    date_part('year', {date_field}) AS year,
                    '{doc_type.name}' AS type,
                    count(*) AS count
                FROM {doc_type.name}, ownable
                WHERE {doc_type.name}.id = ANY($1)
                AND ownable.owner_id = ANY($2)
                AND ownable.{doc_type.name}_id={doc_type.name}.id
                GROUP BY owner_id, date_part('year', {date_field})
            """

        report_query = " UNION ALL ".join(
            [get_report_query(doc_type) for doc_type in self.document_types]
        )
        async with prisma_context(300) as db:
            report = await db.query_raw(report_query, document_ids, owner_ids)

        result_map = {
            type: {
                k: {
                    v["year"]: CountByYear(**v)
                    for v in vals
                    if type == type.all or v["type"] == type.name
                }
                for k, vals in group_by(report, "id").items()
            }
            for type in [DocType.all, *self.document_types]
        }
        return {
            type.name: {
                k: [
                    vals.get(y) or CountByYear(year=y, count=0, type=type.name)
                    for y in range(start_year, MAX_DATA_YEAR)
                ]
                for k, vals in map.items()
            }
            for type, map in result_map.items()
        }

    def _get_fields(
        self,
        start_year: int,
    ) -> list[str]:
        """
        Get fields for the company report query
        """
        current_year = date.today().year

        return [
            "owner.id AS id",
            "owner.name AS name",
            "(COALESCE(acquisition.count, 0) > 0 AND owner.owner_type in ('INDUSTRY', 'INDUSTRY_LARGE')) AS is_acquirer",
            "(COALESCE(acquisition.count, 0) = 0 AND owner.owner_type='INDUSTRY') AS is_competition",
            "financials.symbol AS symbol",
            "ARRAY_AGG(top_docs.id) AS ids",
            f"""
            ARRAY_AGG(
                JSON_BUILD_OBJECT(
                    'url', url,
                    'title', title
                )
            ) AS urls
            """,
            "ARRAY_AGG(top_docs.year) AS years",
            "COUNT(title) AS count",
            "ARRAY_AGG(title) AS titles",
            f"MIN({current_year}-year)::int AS min_age",
            f"ROUND(AVG({current_year}-year)) AS avg_age",
            f"ROUND((1 - (AVG(top_docs.vector) <=> owner.vector))::numeric, 2) AS wheelhouse_score",
            f"ROUND(AVG(relevance_score), 2) AS relevance_score",
            f"""
            ROUND(
                SUM(
                    relevance_score
                    * POW(
                        GREATEST(0.0, ((year - {start_year}) / 24.0)),
                        {self.recency_decay_factor}
                    )
                )::numeric, 2
            ) AS score
            """,
        ]

    async def _fetch_companies(
        self,
        p: CompanyFinderParams,
        description: str,
    ) -> list[CompanyRecord]:
        def by_company_query(inner_query: str) -> str:
            fields = self._get_fields(p.start_year)

            ownable_join = " OR ".join(
                [
                    f"ownable.{doc_type.name}_id=top_docs.id"
                    for doc_type in self.document_types
                ]
            )
            return f"""
                SELECT {', '.join(fields)}
                FROM ({inner_query}) top_docs
                JOIN ownable ON {ownable_join}
                JOIN owner ON owner.id=ownable.owner_id
                LEFT JOIN financials ON financials.owner_id=owner.id
                LEFT JOIN (
                    SELECT owner_id, count(*) AS count
                    FROM acquisition
                    GROUP BY owner_id
                ) acquisition ON acquisition.owner_id=owner.id
                WHERE owner.vector IS NOT null
                GROUP BY owner.name, owner.id, owner.owner_type, financials.symbol, acquisition.count
            """

        companies = await self.get_top_docs(
            description,
            VectorSearchParams(k=p.k, start_year=p.start_year),
            get_query=by_company_query,
            Schema=CompanyRecord,
        )

        owner_ids = tuple([c.id for c in companies])
        document_ids = tuple(flatten([c.ids for c in companies]))
        report_map = await self._fetch_company_reports(
            document_ids, owner_ids, p.start_year
        )

        companies = [
            CompanyRecord(
                **omit(c.model_dump(), "activity", "count_by_year"),
                activity=[v.count for v in report_map["all"][c.id]],
                count_by_year={
                    t.name: report_map[t.name][c.id] for t in self.document_types
                },
            )
            for c in companies
        ]

        return sorted(companies, key=lambda x: x.score, reverse=True)

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

    async def __call__(self, p: CompanyFinderParams) -> FindCompanyResult:
        """
        A specific method to find potential buyers for IP / companies

        Does a cosine similarity search on the description and returns the top k
            with scoring based on recency and relevance
        """
        start = time.monotonic()

        companies = await self._fetch_companies(p, p.description)
        exit_score, competition_score = self._compute_scores(companies)

        logger.info(
            "Find took %s seconds (%s)",
            round(time.monotonic() - start, 2),
            len(companies),
        )

        return FindCompanyResult(
            companies=companies,
            description=p.description,
            exit_score=exit_score,
            competition_score=competition_score,
        )
