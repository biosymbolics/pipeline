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
from core.ner.spacy import get_transformer_nlp
from typings.documents.common import DocType
from utils.string import get_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_YEAR = 2000
RECENCY_DECAY_FACTOR = 2
DEFAULT_K = 1000
SIMILARITY_EXAGGERATION_FACTOR = 50  # adjust with MIN_RELEVANCE_SCORE

ResultSchema = TypeVar("ResultSchema", bound=BaseModel)


class VectorReportClient:
    def __init__(
        self,
        min_year: int = MIN_YEAR,
        recency_decay_factor: int = RECENCY_DECAY_FACTOR,
        exaggeration_factor: int = SIMILARITY_EXAGGERATION_FACTOR,
        document_type: DocType = DocType.patent,
    ):
        self.gpt_client = GptApiClient()
        self.min_year = min_year
        self.recency_decay_factor = recency_decay_factor
        self.exaggeration_factor = exaggeration_factor
        self.document_type = document_type

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

    @staticmethod
    async def _form_vector(
        description: str | None, companies: Sequence[str]
    ) -> list[float]:
        """
        Form query vector for a description & list of companies
        """
        nlp = get_transformer_nlp()

        vectors: list[list[float]] = []

        if description is not None:
            description_doc = nlp(description)
            vectors.append(description_doc.vector.tolist())

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
        self, vector: list[float], k: int, alpha: float = 0.85
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
            query = f"""
                SELECT
                    id,
                    1 - (vector <=> '{vector}') as similarity
                FROM {self.document_type.name}
                ORDER BY vector <-> '{vector}' ASC
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

        inner_query = f"""
            SELECT
                MAX(id) as id,
                AVG(relevance_score) as relevance_score,
                MAX(title) as title,
                AVG(vector) as vector,
                MIN(year) as year
            FROM (
                SELECT
                    id,
                    family_id,
                    POW((1 - (vector <=> '{vector}')), {self.exaggeration_factor})::numeric as relevance_score,
                    title,
                    vector,
                    date_part('year', priority_date) as year
                FROM {self.document_type.name}
                WHERE id = ANY($1)
            ) s
            GROUP BY s.family_id
        """
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
