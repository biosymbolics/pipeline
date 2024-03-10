import time
from prisma import Prisma
import torch
import logging

from clients.low_level.prisma import prisma_client
from core.vector.vectorizer import Vectorizer
from typings.documents.common import MentionCandidate
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

        float_vec = mention_vec.tolist()

        # ALTER SYSTEM SET pg_trgm.similarity_threshold = 0.9;
        query = f"""
            SELECT
                umls.id AS id,
                umls.name as name,
                synonyms,
                type_ids AS types,
                COALESCE(1 - (vector <=> '{float_vec}'), 0.0) AS semantic_similarity,
                similarity(matches.name, '{mention}') AS syntactic_similarity,
                vector::text AS vector
            FROM umls
            JOIN (
                SELECT * FROM (
                    SELECT id, name
                    FROM umls
                    WHERE (vector <=> '{float_vec}') < 1 - {min_similarity}
                    ORDER BY vector <=> '{float_vec}' ASC
                    LIMIT {k}
                ) s

                UNION

                SELECT umls_id AS id, term AS name
                FROM umls_synonym
                WHERE '{mention}' % term
            ) AS matches ON matches.id = umls.id
            WHERE vector is not null
        """
        start = time.time()

        try:
            res = await self.db.query_raw(query)
        except Exception as e:
            logger.error(f"Error querying for {mention}: {e} ({query})")
            raise e
        logger.info("Query time: %ss", round(time.time() - start))
        return [MentionCandidate(**r) for r in res], mention_vec

    async def __call__(
        self,
        mentions: list[str],
        vectors: list[torch.Tensor] | None = None,
        k: int = 10,
        min_similarity: float = 0.85,
    ) -> list[tuple[list[MentionCandidate], torch.Tensor]]:
        """
        Generate candidates for a list of mentions
        """
        _vectors = vectors or self.vectorizer.vectorize(mentions)
        mention_vecs = [l1_regularize(v) for v in _vectors]

        candidates = [
            await self.get_candidates(mention, vec, k, min_similarity)
            for mention, vec in zip(mentions, mention_vecs)
        ]
        return candidates
