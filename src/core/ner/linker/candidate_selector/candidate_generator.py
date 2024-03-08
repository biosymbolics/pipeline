import torch
import logging

from clients.low_level.prisma import prisma_client
from core.vector.vectorizer import Vectorizer
from typings.documents.common import MentionCandidate
from utils.tensor import l1_regularize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateGenerator:
    def __init__(self):
        self.db = None
        self.vectorizer = Vectorizer()

    async def get_candidates(
        self, mention_vec: torch.Tensor, k: int, min_similarity: float = 0.85
    ) -> list[MentionCandidate]:
        """
        Get candidates for a mention
        """
        if self.db is None:
            self.db = await prisma_client(300)
            assert self.db is not None

        float_vec = mention_vec.tolist()

        query = f"""
            SELECT * FROM (
                SELECT
                    id,
                    name,
                    synonyms,
                    type_ids AS types,
                    1 - (vector <=> '{float_vec}') as similarity
                FROM umls
                ORDER BY (vector <=> '{float_vec}') ASC
                LIMIT {k}
            ) s
            WHERE similarity >= {min_similarity}
        """
        res = await self.db.query_raw(query)
        return [MentionCandidate(**r) for r in res]

    async def __call__(
        self, mentions: list[str], k: int = 5, min_similarity: float = 0.85
    ) -> list[list[MentionCandidate]]:
        """
        Generate candidates for a list of mentions
        """

        mention_vecs = [l1_regularize(v) for v in self.vectorizer.vectorize(mentions)]
        candidates = [
            await self.get_candidates(vec, k, min_similarity) for vec in mention_vecs
        ]
        return candidates
