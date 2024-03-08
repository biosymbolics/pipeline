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
        self.vectorizer = Vectorizer.get_instance()

    async def get_candidates(
        self,
        mention: str,
        mention_vec: torch.Tensor,
        k: int,
        min_similarity: float = 0.85,
    ) -> list[MentionCandidate]:
        """
        Get candidates for a mention
        """
        if self.db is None:
            self.db = await prisma_client(300)
            assert self.db is not None

        float_vec = mention_vec.tolist()

        # ALTER SYSTEM SET pg_trgm.similarity_threshold = 0.9;
        query = f"""
            SELECT
                umls.id as id,
                name,
                ARRAY_AGG(umls_synonym.term) as synonyms,
                MAX(type_ids) AS types,
                1 - (AVG(vector) <=> '{float_vec}') as similarity,
                MAX(similarity('{mention}', term)) as syntactic_similarity
            FROM umls_synonym, umls
            JOIN (
                SELECT * from (
                    SELECT id
                    FROM umls
                    WHERE (vector <=> '{float_vec}') < 1 - {min_similarity}
                    ORDER BY vector <=> '{float_vec}' ASC
                    LIMIT {k}
                ) s

                UNION

                select umls_id as id from umls_synonym where '{mention}' % term
            ) AS matches ON matches.id = umls.id
            WHERE umls.id = umls_synonym.umls_id
            GROUP BY umls.id, name
            ORDER BY vector <=> '{float_vec}' ASC
            LIMIT {k}
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
            await self.get_candidates(mention, vec, k, min_similarity)
            for mention, vec in zip(mentions, mention_vecs)
        ]
        return candidates
