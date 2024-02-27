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
from typings.client import CompanyFinderParams
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
        self, document_ids: Sequence[str], owner_ids: Sequence[int]
    ) -> dict[int, list[CountByYear]]:
        """
        Fetches company reports for a set of document and owner ids

        Returns map between owner_id and list of corresponding list[CountByYear]
        """

        def get_report_query(doc_type: DocType):
            date_field = DOC_TYPE_DATE_MAP[doc_type]
            return f"""
                SELECT
                    owner_id as id,
                    date_part('year', {date_field}) as year,
                    count(*) as count
                FROM {doc_type.name}, ownable
                WHERE {doc_type.name}.id = ANY($1)
                AND ownable.owner_id = ANY($2)
                AND ownable.{doc_type.name}_id={doc_type.name}.id
                AND date_part('year', {date_field}) >= {self.min_year}
                GROUP BY owner_id, date_part('year', {date_field})
            """

        report_query = " UNION ALL ".join(
            [get_report_query(doc_type) for doc_type in self.document_types]
        )
        async with prisma_context(300) as db:
            report = await db.query_raw(report_query, document_ids, owner_ids)

        result_map = {
            k: {v["year"]: CountByYear(**v) for v in vals}
            for k, vals in group_by(report, "id").items()
        }
        return {
            k: [
                vals.get(y) or CountByYear(year=y, count=0)
                for y in range(self.min_year, MAX_DATA_YEAR)
            ]
            for k, vals in result_map.items()
        }

    def _get_fields(
        self,
        companies: Sequence[str],
        description: str | None,
        vector: Sequence[float],
    ) -> list[str]:
        """
        Get fields for the query that differ based on the presence of companies and description
        """
        current_year = date.today().year

        common_fields = [
            "owner.id AS id",
            "owner.name AS name",
            "(owner.acquisition_count > 0 AND owner.owner_type in ('INDUSTRY', 'INDUSTRY_LARGE')) AS is_acquirer",
            "(owner.acquisition_count = 0 AND owner.owner_type='INDUSTRY') AS is_competition",
            "financials.symbol AS symbol",
            "ARRAY_AGG(top_docs.id) AS ids",
            "COUNT(title) AS count",
            "ARRAY_AGG(title) AS titles",
            f"MIN({current_year}-year)::int AS min_age",
            f"ROUND(AVG({current_year}-year)) AS avg_age",
            f"ROUND((AVG(top_docs.vector) <-> owner.vector)::numeric * 10, 2) AS wheelhouse_score",
        ]
        company_fields = [
            f"ROUND((1 / exp(vector <-> '{vector}'))::numeric * 10, 2) AS relevance_score",
            f"ROUND((1 / exp(vector <-> '{vector}'))::numeric * 10, 2) AS score",
        ]

        description_fields = [
            f"ROUND(AVG(relevance_score), 2) AS relevance_score",
            f"""
            ROUND(
                SUM(
                    relevance_score
                    * POW(
                        GREATEST(0.0, ((year - {self.min_year}) / 24)),
                        {self.recency_decay_factor}
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

    async def _fetch_companies(
        self,
        p: CompanyFinderParams,
        description: str | None,
        vector: list[float],
    ) -> list[CompanyRecord]:
        def by_company_query(inner_query: str) -> str:
            fields = self._get_fields(p.similar_companies, description, vector)

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
                WHERE owner.vector is not null
                GROUP BY owner.name, owner.id, owner.owner_type, financials.symbol
            """

        companies = await self.get_top_docs(
            description,
            p.similar_companies,
            p.k,
            get_query=by_company_query,
            Schema=CompanyRecord,
        )

        owner_ids = tuple([c.id for c in companies])
        document_ids = tuple(flatten([c.ids for c in companies]))
        report_map = await self._fetch_company_reports(document_ids, owner_ids)

        companies = [
            CompanyRecord(
                **omit(c.model_dump(), "activity", "count_by_year"),
                activity=[v.count for v in report_map[c.id]],
                count_by_year=report_map[c.id],
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

        Args:
            p.description: description of the IP
            p.companies: list of companies to search
            p.k: number of nearest neighbors to consider
            p.use_gpt_expansion: whether to use GPT to expand the description (mostly for testing)
        """
        start = time.monotonic()

        if p.use_gpt_expansion and p.description is not None:
            logger.info("Using GPT to expand description")
            description = await self.gpt_client.generate_ip_description(p.description)
        else:
            description = p.description

        vector = await self._form_vector(description, p.similar_companies)
        companies = await self._fetch_companies(p, description, vector)
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
