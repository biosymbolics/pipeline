import json
from typing import AsyncIterable
from prisma import Prisma
import torch
import logging

from clients.low_level.prisma import prisma_client
from nlp.vectorizing.vectorizer import Vectorizer
from typings.documents.common import MentionCandidate
from utils.async_utils import gather_with_concurrency_limit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateGenerator:
    def __init__(self, db: Prisma):
        """
        Initialize candidate generator

        Use `create` to instantiate with async dependencies
        """
        self.db = db
        self.vectorizer = Vectorizer.get_instance()

    @staticmethod
    async def create() -> "CandidateGenerator":
        db = await prisma_client(300)
        return CandidateGenerator(db)

    async def get_candidates(
        self,
        mention: str | None,
        mention_vec: torch.Tensor,
        k: int,
        min_similarity: float = 0.85,
    ) -> tuple[list[MentionCandidate], torch.Tensor]:
        """
        Get candidates for a mention
        """
        max_distance = 1 - min_similarity

        # ALTER SYSTEM SET pg_trgm.similarity_threshold = 0.9;
        query = f"""
            SELECT
                umls.id AS id,
                umls.name AS name,
                synonyms,
                type_ids AS types,
                COALESCE(1 - (vector <=> $1::vector), 0.0) AS semantic_similarity,
                {'MAX(similarity($2, term))' if mention else '1'} AS syntactic_similarity,
                vector::text AS vector
            FROM umls_synonym, umls
            JOIN (
                SELECT * FROM (
                    SELECT id, name
                    FROM umls
                    WHERE (vector <=> $1::vector) < {max_distance}
                    -- AND is_eligible=true
                    ORDER BY vector <=> $1::vector ASC
                    LIMIT {k}
                ) s

                {'UNION SELECT distinct umls_id AS id, term AS name FROM umls_synonym WHERE term % $2' if mention else ''}

                LIMIT {k}
            ) AS matches ON matches.id = umls.id
            WHERE umls.id = umls_synonym.umls_id
            GROUP BY umls.id, umls.name, synonyms, types, vector
            -- AND is_eligible=true
        """
        try:
            params = (
                [json.dumps(mention_vec.tolist()), mention]
                if mention
                else [json.dumps(mention_vec.tolist())]
            )
            res = await self.db.query_raw(query, *params)
        except Exception:
            logger.exception("Failed to query for candidates")
            return [], mention_vec
        return [MentionCandidate(**r) for r in res], mention_vec

    async def __call__(
        self,
        mentions: list[str] | None,
        vectors: list[torch.Tensor] | None = None,
        k: int = 10,
        min_similarity: float = 0.85,
    ) -> AsyncIterable[tuple[list[MentionCandidate], torch.Tensor]]:
        """
        Generate candidates for a list of mentions
        """
        if not mentions and not vectors:
            raise ValueError("Must provide either mentions or vectors")

        mention_vecs = vectors or self.vectorizer.vectorize(mentions or [])
        mention_texts = mentions or [None] * len(mention_vecs)

        candidate_sets = await gather_with_concurrency_limit(
            10,
            *[
                self.get_candidates(mention, vec, k, min_similarity)
                for mention, vec in zip(mention_texts, mention_vecs)
            ],
        )
        for candidates, vec in candidate_sets:
            yield candidates, vec
