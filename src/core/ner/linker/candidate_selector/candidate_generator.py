from typing import AsyncIterable
from prisma import Prisma
import torch
import logging

from clients.low_level.prisma import prisma_client
from core.vector.vectorizer import Vectorizer
from typings.documents.common import MentionCandidate
from utils.async_utils import gather_with_concurrency_limit
from utils.tensor import l1_regularize

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
        mention: str,
        mention_vec: torch.Tensor | None,
        k: int,
        min_similarity: float = 0.85,
    ) -> tuple[list[MentionCandidate], torch.Tensor]:
        """
        Get candidates for a mention
        """
        if mention_vec is None:
            logger.warning("mention_vec is None, one-off vectorizing mention (slow!)")
            mention_vec = self.vectorizer.vectorize([mention])[0]

        if len(mention_vec.shape) == 0:
            logger.warning("mention_vec is empty (%s)", mention_vec.shape)
            return [], mention_vec

        max_distance = 1 - min_similarity

        # ALTER SYSTEM SET pg_trgm.similarity_threshold = 0.9;
        query = f"""
            SELECT
                umls.id AS id,
                umls.name AS name,
                synonyms,
                type_ids AS types,
                COALESCE(1 - (vector <=> $2::vector), 0.0) AS semantic_similarity,
                similarity(matches.name, $1) AS syntactic_similarity,
                vector::text AS vector
            FROM umls
            JOIN (
                SELECT * FROM (
                    SELECT id, name
                    FROM umls
                    WHERE (vector <=> $2::vector) < {max_distance}
                    AND is_eligible=true
                    ORDER BY vector <=> $2::vector ASC
                    LIMIT {k}
                ) s

                UNION

                SELECT distinct umls_id AS id, term AS name
                FROM umls_synonym
                WHERE term % $1

                LIMIT {k}
            ) AS matches ON matches.id = umls.id
            -- WHERE is_eligible=true
        """
        try:
            res = await self.db.query_raw(query, mention, mention_vec.tolist())
        except Exception:
            logger.exception("Failed to query for candidates")
            return [], mention_vec
        return [MentionCandidate(**r) for r in res], mention_vec

    async def __call__(
        self,
        mentions: list[str],
        vectors: list[torch.Tensor] | None = None,
        k: int = 10,
        min_similarity: float = 0.85,
    ) -> AsyncIterable[tuple[list[MentionCandidate], torch.Tensor]]:
        """
        Generate candidates for a list of mentions
        """
        _vectors = vectors or self.vectorizer.vectorize(mentions)
        mention_vecs = [l1_regularize(v) for v in _vectors]

        candidate_sets = gather_with_concurrency_limit(
            10,
            *[
                self.get_candidates(mention, vec, k, min_similarity)
                for mention, vec in zip(mentions, mention_vecs)
            ],
        )
        for candidates, vec in await candidate_sets:
            yield candidates, vec
