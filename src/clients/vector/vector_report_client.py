"""
Base semantic finder
"""

import json
import logging
from typing import Callable, Sequence, Type, TypeVar
from pydantic import BaseModel
import torch

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from clients.openai.gpt_client import GptApiClient
from clients.vector.types import TopDocRecord, TopDocsByYear
from core.vector import Vectorizer
from typings.documents.common import DOC_TYPE_DATE_MAP, DOC_TYPE_DEDUP_ID_MAP, DocType
from utils.string import get_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_YEAR = 2000
RECENCY_DECAY_FACTOR = 2
DEFAULT_K = 1000

ResultSchema = TypeVar("ResultSchema", bound=BaseModel)


class VectorReportClient:
    def __init__(
        self,
        min_year: int = MIN_YEAR,
        recency_decay_factor: int = RECENCY_DECAY_FACTOR,
        document_types: Sequence[DocType] = [DocType.patent, DocType.trial],
    ):
        self.document_types = document_types
        self.gpt_client = GptApiClient()
        self.min_year = min_year
        self.recency_decay_factor = recency_decay_factor
        self.vectorizer = Vectorizer()

    @staticmethod
    async def _get_companies_vector(companies: Sequence[str]) -> list[float]:
        """
        From the avg vector for a list of companies
        """
        query = f"""
            SELECT AVG(vector)::text vector FROM owner
            WHERE name=ANY($1)
        """
        async with prisma_context(300) as db:
            result = await db.query_raw(query, companies)

        return json.loads(result[0]["vector"])

    async def _form_vector(
        self, description: str | None, companies: Sequence[str]
    ) -> list[float]:
        """
        Form query vector for a description & list of companies
        """
        vectors: list[list[float]] = []

        if description is not None:
            desc_vector = self.vectorizer(description).tolist()
            vectors.append(desc_vector)

        if len(companies) > 0:
            company_vector = await VectorReportClient._get_companies_vector(companies)
            vectors.append(company_vector)

        combined_vector: list[float] = (
            torch.stack([torch.tensor(v) for v in vectors]).mean(dim=0).tolist()
        )

        return combined_vector

    def _calc_min_similarity(
        self, similarities: Sequence[float], alpha: float
    ) -> float:
        """
        Get the default min similarity score
        min=avg(sim)+α*σ(sim)

        Args:
            similarities (Sequence[float]): similarity scores
            alpha (float): alpha for the calculation

        Returns:
            min similarity score (float)

        Taken from https://www.researchgate.net/post/Determination-of-threshold-for-cosine-similarity-score
        """
        mean = sum(similarities) / len(similarities)
        stddev = (
            sum((score - mean) ** 2 for score in similarities) / len(similarities)
        ) ** 0.5
        return mean + alpha * stddev

    async def _get_top_ids(
        self, vector: list[float], k: int, alpha: float = 0.70
    ) -> list[str]:
        """
        Get the ids of the k nearest neighbors to a vector
        Returns ids only above a certain threshold, determined dynamically.
        Cached.

        Args:
            vector (list[float]): vector to search
            k (int): number of nearest neighbors to return
            alpha (float, optional): alpha for min sim score calc. Defaults to 0.75.
        """

        async def fetch():
            def get_doc_query(doc_type: DocType):
                date_field = DOC_TYPE_DATE_MAP[doc_type]
                return f"""
                    SELECT id, vector
                    FROM {doc_type.name}
                    WHERE date_part('year', {date_field}) >= {self.min_year}
                """

            doc_queries = " UNION ALL ".join(
                [get_doc_query(doc_type) for doc_type in self.document_types]
            )

            query = f"""
                SELECT
                    id,
                    1 - (vector <=> '{vector}') as similarity
                FROM ({doc_queries}) docs
                ORDER BY 1 - (vector <=> '{vector}') DESC
                LIMIT {k}
            """
            async with prisma_context(300) as db:
                records = await db.query_raw(query)

            scores = [
                record["similarity"]
                for record in records
                if record["similarity"] is not None  # not sure why it returns null
            ]
            min_similiarity = self._calc_min_similarity(scores, alpha)

            above_threshold_ids = [
                record["id"]
                for record in records
                if (record["similarity"] or 0) >= min_similiarity
            ]

            logger.info(
                "Returning %s ids above similarity threshold %s",
                len(above_threshold_ids),
                min_similiarity,
            )

            return above_threshold_ids

        cache_key = get_id([vector, k])
        response = await retrieve_with_cache_check(
            fetch,
            key=cache_key,
            decode=lambda str_data: storage_decoder(str_data),
            use_filesystem=True,
        )

        return response

    async def get_top_docs(
        self,
        description: str | None = None,
        similar_companies: Sequence[str] = [],
        k: int = DEFAULT_K,
        get_query: Callable[[str], str] | None = None,
        Schema: Type[ResultSchema] = TopDocRecord,
    ) -> list[ResultSchema]:
        """
        Get the query to get the top documents for a vector

        Args:
            description (str, optional): description of the concept. Defaults to None.
            similar_companies (Sequence[str], optional): list of similar companies. Defaults to [].
            k (int, optional): number of top documents to return. Defaults to DEFAULT_K.
            get_query (Callable[[str], str], optional): function yielding the overall desired query.
            Schema (Type[ResultSchema], optional): schema to use for the result. Defaults to TopDocRecord.
        """
        vector = await self._form_vector(description, similar_companies)
        ids = await self._get_top_ids(vector, k)

        def get_inner_query(doc_type: DocType):
            date_field = DOC_TYPE_DATE_MAP[doc_type]
            dedup_id_field = DOC_TYPE_DEDUP_ID_MAP[doc_type]
            return f"""
                SELECT
                    MAX(id) as id,
                    AVG(relevance_score) as relevance_score,
                    MAX(title) as title,
                    MAX(url) as url,
                    AVG(vector) as vector,
                    MAX(year) as year
                FROM (
                    SELECT
                        id,
                        {dedup_id_field} as dedup_id,
                        (1 - (vector <=> '{vector}'))::numeric as relevance_score,
                        title,
                        url,
                        vector,
                        date_part('year', {date_field})::int as year
                    FROM {doc_type.name}
                    WHERE id = ANY($1)
                    AND {date_field} is not null
                ) s
                GROUP BY s.dedup_id
            """

        inner_query = " UNION ALL ".join(
            [get_inner_query(doc_type) for doc_type in self.document_types]
        )
        query = get_query(inner_query) if get_query else inner_query

        async with prisma_context(300) as db:
            records: list[dict] = await db.query_raw(query, ids)

        return [Schema(**record) for record in records]

    async def get_top_docs_by_year(
        self, description: str, similar_companies: Sequence[str] = []
    ) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies,
        aggregated by year

        Args:
            description (str): description of the concept
            similar_companies (Sequence[str], optional): list of similar companies. Defaults to [].
        """

        def by_year_query(doc_query: str) -> str:
            return f"""
                SELECT
                    ARRAY_AGG(id) as ids,
                    COUNT(*) as count,
                    AVG(relevance_score) as avg_score,
                    ARRAY_AGG(relevance_score) as scores,
                    ARRAY_AGG(title) as titles,
                    SUM(relevance_score) as total_score,
                    year
                FROM ({doc_query}) top_docs
                GROUP BY year
            """

        return await self.get_top_docs(
            description,
            similar_companies,
            get_query=by_year_query,
            Schema=TopDocsByYear,
        )

    async def __call__(
        self, description: str, similar_companies: Sequence[str] = []
    ) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies, by year

        Args:
            description (str): description of the concept
            similar_companies (Sequence[str], optional): list of similar companies. Defaults to [].
        """
        logger.info(
            "Getting top docs by year for description: '%s' and similar companies: %s",
            description,
            similar_companies,
        )
        return await self.get_top_docs_by_year(description, similar_companies)
