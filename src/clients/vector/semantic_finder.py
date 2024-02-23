"""
Base semantic finder
"""

import json
import logging
from typing import Sequence
import torch

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from clients.openai.gpt_client import GptApiClient
from clients.vector.types import TopDocRecord, TopDocsByYear
from core.ner.spacy import get_transformer_nlp
from utils.string import get_id

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MIN_YEAR = 2000
RECENCY_DECAY_FACTOR = 2
SIMILARITY_EXAGGERATION_FACTOR = 50
MIN_RELEVANCE_SCORE = 0.5


class SemanticFinder:
    def __init__(
        self,
        min_year: int = MIN_YEAR,
        recency_decay_factor: int = RECENCY_DECAY_FACTOR,
        exaggeration_factor: int = SIMILARITY_EXAGGERATION_FACTOR,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.gpt_client = GptApiClient()
        self.min_year = min_year
        self.recency_decay_factor = recency_decay_factor
        self.exaggeration_factor = exaggeration_factor
        self.min_relevance_score = min_relevance_score

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
            company_vector = await SemanticFinder._get_companies_vector(companies)
            vectors.append(company_vector)

        combined_vector: list[float] = (
            torch.stack([torch.tensor(v) for v in vectors]).mean(dim=0).tolist()
        )

        return combined_vector

    def get_top_docs_query(self, vector: list[float]) -> str:
        """
        Get the query to get the top documents for a vector
        """
        query = f"""
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
                FROM patent
                WHERE id = ANY($1)
            ) s
            WHERE relevance_score >= {self.min_relevance_score}
            GROUP BY s.family_id
        """
        return query

    @staticmethod
    async def _get_knn_ids(vector: list[float], k: int) -> list[str]:
        """
        Get the ids of the k nearest neighbors to a vector
        Cached.

        Args:
            vector (list[float]): vector to search
            k (int): number of nearest neighbors to return
        """

        async def fetch():
            query = f"""
                SELECT id FROM patent
                ORDER BY (1 - (vector <=> '{vector}')) DESC LIMIT {k}
            """
            async with prisma_context(300) as db:
                records = await db.query_raw(query)

            return [record["id"] for record in records]

        cache_key = get_id([vector, k])
        response = await retrieve_with_cache_check(
            fetch,
            key=cache_key,
            decode=lambda str_data: storage_decoder(str_data),
            use_filesystem=True,
        )

        return response

    async def get_top_docs(
        self, description: str, similar_companies: Sequence[str] = []
    ) -> list[TopDocRecord]:
        """
        Get the top documents for a description and (optionally) similar companies
        (unaggregated)

        Args:
            description (str): description of the concept
            similar_companies (Sequence[str], optional): list of similar companies. Defaults to [].
        """
        vector = await self._form_vector(description, similar_companies)
        async with prisma_context(300) as db:
            top_docs = await db.query_raw(self.get_top_docs_query(vector))

        return [TopDocRecord(**record) for record in top_docs]

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
        vector = await self._form_vector(description, similar_companies)
        doc_query = self.get_top_docs_query(vector)
        query = f"""
            SELECT
                ARRAY_AGG(id) as ids,
                AVG(relevance_score) as avg_score,
                ARRAY_AGG(relevance_score) as scores,
                ARRAY_AGG(title) as titles,
                year
            FROM ({doc_query}) top_docs
            GROUP BY year
        """
        async with prisma_context(300) as db:
            top_docs = await db.query_raw(query)

        return [TopDocsByYear(**record) for record in top_docs]

    async def __call__(
        self, description: str, similar_companies: Sequence[str] = []
    ) -> list[TopDocsByYear]:
        """
        Get the top documents for a description and (optionally) similar companies, by year

        Args:
            description (str): description of the concept
            similar_companies (Sequence[str], optional): list of similar companies. Defaults to [].
        """
        return await self.get_top_docs_by_year(description, similar_companies)
