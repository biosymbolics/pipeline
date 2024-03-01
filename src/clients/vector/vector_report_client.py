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
from core.vector import Vectorizer
from typings.documents.common import DOC_TYPE_DATE_MAP, DOC_TYPE_DEDUP_ID_MAP, DocType
from utils.string import get_id

from .types import TopDocRecord, TopDocsByYear, VectorSearchParams, VectorSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RECENCY_DECAY_FACTOR = 2

ResultSchema = TypeVar("ResultSchema", bound=BaseModel)


class VectorReportClient:
    def __init__(
        self,
        recency_decay_factor: int = RECENCY_DECAY_FACTOR,
        document_types: Sequence[DocType] = [DocType.patent, DocType.trial],
    ):
        self.document_types = document_types
        self.gpt_client = GptApiClient()
        self.recency_decay_factor = recency_decay_factor
        self.vectorizer = Vectorizer()

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

    async def get_top_doc_ids(
        self,
        search_params: VectorSearchParams,
    ) -> list[str]:
        """
        Get the ids of the k nearest neighbors to a vector
        Returns ids only above a certain threshold, determined dynamically.
        Cached.

        Args:
            search_params (VectorSearchParams): search parameters
            alpha (float, optional): alpha for min sim score calc. Defaults to 0.75.
        """

        async def fetch():
            def get_doc_query(doc_type: DocType):
                date_field = DOC_TYPE_DATE_MAP[doc_type]
                return f"""
                    SELECT id, vector
                    FROM {doc_type.name}
                    WHERE date_part('year', {date_field}) >= {search_params.min_year}
                    AND NOT id = ANY($1)
                """

            doc_queries = " UNION ALL ".join(
                [get_doc_query(doc_type) for doc_type in self.document_types]
            )

            query = f"""
                SELECT
                    id,
                    1 - (vector <=> '{search_params.vector}') as similarity
                FROM ({doc_queries}) docs
                ORDER BY (vector <=> '{search_params.vector}') ASC
                LIMIT {search_params.k}
            """
            async with prisma_context(300) as db:
                records = await db.query_raw(query, search_params.skip_ids)

            scores = [
                record["similarity"]
                for record in records
                if record["similarity"] is not None  # not sure why it returns null
            ]
            min_similiarity = self._calc_min_similarity(scores, search_params.alpha)

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

        cache_key = get_id([search_params.vector, search_params.k])
        response = await retrieve_with_cache_check(
            fetch,
            key=cache_key,
            decode=lambda str_data: storage_decoder(str_data),
            use_filesystem=True,
        )

        return response

    async def get_top_docs_by_vector(
        self,
        search_params: VectorSearchParams,
        get_query: Callable[[str], str] | None = None,
        Schema: Type[ResultSchema] = TopDocRecord,
    ):
        """
        Get the top documents for a vector

        Args:
            search_params (VectorSearchParams): search parameters
            get_query (Callable[[str], str], optional): function yielding the overall desired query.
            Schema (Type[ResultSchema], optional): schema to use for the result. Defaults to TopDocRecord.
        """

        def get_inner_query(doc_type: DocType):
            date_field = DOC_TYPE_DATE_MAP[doc_type]
            dedup_id_field = DOC_TYPE_DEDUP_ID_MAP[doc_type]
            return f"""
                SELECT
                    MAX(id) AS id,
                    AVG(relevance_score) AS relevance_score,
                    MAX(title) AS title,
                    CONCAT(MAX(title), ': ', MAX(LEFT(abstract, 150)), '...') AS description,
                    MAX(url) AS url,
                    -- if no get_query, we cast vector to string (because prisma can't deserialize vectors)
                    AVG(vector){'::text' if get_query is None else ''} AS vector,
                    MAX(year) AS year,
                    '{doc_type.name}' AS type
                FROM (
                    SELECT
                        id,
                        {dedup_id_field} AS dedup_id,
                        (1 - (vector <=> '{search_params.vector}'))::numeric AS relevance_score,
                        abstract,
                        title,
                        url,
                        vector,
                        date_part('year', {date_field})::int AS year
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
        ids = await self.get_top_doc_ids(search_params)

        async with prisma_context(300) as db:
            records: list[dict] = await db.query_raw(query, ids)

        return [Schema(**record) for record in records]

    async def get_top_docs(
        self,
        description: str,
        search_params: VectorSearchParams = VectorSearchParams(),
        get_query: Callable[[str], str] | None = None,
        Schema: Type[ResultSchema] = TopDocRecord,
    ) -> list[ResultSchema]:
        """
        Get the query to get the top documents for a vector

        Args:
            description (str, optional): description of the concept. Defaults to None.
            search_params (VectorSearchParams, optional): search parameters.
            get_query (Callable[[str], str], optional): function yielding the overall desired query.
            Schema (Type[ResultSchema], optional): schema to use for the result. Defaults to TopDocRecord.
        """
        vector = self.vectorizer(description).tolist()
        top_docs = await self.get_top_docs_by_vector(
            search_params.merge({"vector": vector}),
            get_query=get_query,
            Schema=Schema,
        )
        return top_docs

    async def get_top_docs_by_year_and_vector(
        self, search_params: VectorSearchParams
    ) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies,
        aggregated by year.

        Args:
            description (str): description of the concept
        """

        def by_year_query(doc_query: str) -> str:
            return f"""
                SELECT
                    ARRAY_AGG(id) as ids,
                    COUNT(*) as count,
                    AVG(relevance_score) as avg_score,
                    ARRAY_AGG(relevance_score) as scores,
                    ARRAY_AGG(title) as titles,
                    ARRAY_AGG(distinct description) as descriptions,
                    SUM(relevance_score) as total_score,
                    year
                FROM ({doc_query}) top_docs
                GROUP BY year
            """

        return await self.get_top_docs_by_vector(
            search_params,
            get_query=by_year_query,
            Schema=TopDocsByYear,
        )

    async def get_top_docs_by_year(
        self,
        description: str,
        search_params: VectorSearchParams = VectorSearchParams(),
    ) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies, aggregated by year
        """
        vector = self.vectorizer(description).tolist()
        return await self.get_top_docs_by_year_and_vector(
            search_params.merge({"vector": vector})
        )

    async def __call__(self, description: str, **kwargs) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies, by year

        Args:
            description (str): description of the concept
        """
        search_params = VectorSearchParams(**kwargs)
        logger.info(
            "Getting top docs by year for description: '%s', params: %s",
            description,
            search_params,
        )
        return await self.get_top_docs_by_year(description, search_params)
