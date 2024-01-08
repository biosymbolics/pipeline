import time
from typing import Sequence
from annoy import AnnoyIndex
from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
)
import torch
import logging

from core.ner.spacy import get_transformer_nlp
from core.ner.types import CanonicalEntity, DocEntity
from constants.umls import PREFERRED_UMLS_TYPES
from utils.classes import overrides

from .types import AbstractCandidateSelector, EntityScoreVector
from .utils import (
    score_semantic_candidate as score_candidate,
    l1_regularize,
    apply_umls_word_overrides,
    candidate_to_canonical,
)

DEFAULT_K = 20
MIN_SIMILARITY = 1.1
UMLS_KB = None
DOC_VECTOR_WEIGHT = 0.2
WORD_EMBEDDING_LENGTH = 768


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticCandidateSelector(AbstractCandidateSelector):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    Uses semantic comparison (e.g. word vector cosine similarity) to select the best candidate.
    """

    def __init__(
        self,
        *args,
        min_similarity: float = MIN_SIMILARITY,
        vector_length: int = WORD_EMBEDDING_LENGTH,
        **kwargs
    ):
        global UMLS_KB

        if UMLS_KB is None:
            from scispacy.linking_utils import UmlsKnowledgeBase

            UMLS_KB = UmlsKnowledgeBase()

        self.kb = UMLS_KB
        self.candidate_generator = CandidateGenerator(*args, kb=UMLS_KB, **kwargs)
        self.min_similarity = min_similarity
        self.vector_length = vector_length
        self.nlp = get_transformer_nlp()

    def batch_vectorize(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """
        Vectorize a text
        """
        docs = list(self.nlp.pipe(texts))
        bert_vecs = [
            l1_regularize(torch.tensor(doc.vector))
            if len(doc.vector) > 0
            else torch.zeros(self.vector_length)
            for doc in docs
        ]
        return bert_vecs

    def _get_candidates(self, text: str) -> list[MentionCandidate]:
        """
        Wrapper around candidate generator call that handles word overrides
        """
        candidates = self.candidate_generator([text], k=DEFAULT_K)[0]
        with_overrides = apply_umls_word_overrides(text, candidates)
        return with_overrides

    def create_ann_index(
        self,
        candidates: Sequence[MentionCandidate],
    ) -> AnnoyIndex:
        """
        Create an Annoy index for a list of candidates
        """
        start = time.monotonic()
        # other metrics don't work well
        umls_ann = AnnoyIndex(self.vector_length, metric="angular")

        canonical_names = [
            self.kb.cui_to_entity[c.concept_id].canonical_name.lower()
            for c in candidates
        ]
        # only use the first alias, since that's the most syntactically similar to the mention
        tuis = [self.kb.cui_to_entity[c.concept_id].types[0] for c in candidates]
        types = [PREFERRED_UMLS_TYPES.get(tui) or tui for tui in tuis]

        vectors = self.batch_vectorize(canonical_names + types)
        cn_vectors = vectors[0 : len(canonical_names)]
        type_vectors = vectors[len(canonical_names) :]

        for i in range(len(candidates)):
            combined_vec = (0.8 * cn_vectors[i]) + (0.2 * type_vectors[i])
            umls_ann.add_item(i, combined_vec.detach().cpu().numpy())

        umls_ann.build(len(candidates))
        logger.debug("Took %s seconds to build Annoy index", time.monotonic() - start)
        return umls_ann

    def _score_candidate(
        self,
        concept_id: str,
        mention_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        syntactic_similarity: float,
        semantic_distance: float,
    ) -> float:
        return score_candidate(
            concept_id,
            self.kb.cui_to_entity[concept_id].canonical_name,
            self.kb.cui_to_entity[concept_id].types,
            mention_vector,
            candidate_vector=candidate_vector,
            syntactic_similarity=syntactic_similarity,
            semantic_distance=semantic_distance,
        )

    def _get_best_canonical(
        self, vector: torch.Tensor, candidates: Sequence[MentionCandidate]
    ) -> EntityScoreVector | None:
        """
        Get best candidate by semantic similarity
        """
        if len(candidates) == 0:
            logger.warning("get_best_canoical called with no candidates")
            return None

        if len(vector) == 0:
            logger.warning(
                "No vector for %s, probably OOD (%s)",
                candidates[0].aliases[0],
                vector,
            )
            return None

        umls_ann = self.create_ann_index(candidates)

        ids, distances = umls_ann.get_nns_by_vector(
            vector.tolist(), 10, search_k=-1, include_distances=True
        )

        if len(ids) == 0:
            logger.warning("No candidates for %s", candidates[0].aliases[0])
            return None

        scored_candidates = sorted(
            [
                (
                    candidates[id],
                    self._score_candidate(
                        candidates[id].concept_id,
                        vector,
                        candidate_vector=torch.tensor(umls_ann.get_item_vector(id)),
                        syntactic_similarity=candidates[id].similarities[0],
                        semantic_distance=distances[i],
                    ),
                    umls_ann.get_item_vector(id),
                )
                for i, id in enumerate(ids)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidate = scored_candidates[0][0]
        top_score = scored_candidates[0][1]
        top_vector = scored_candidates[0][2]

        print([(c[0].aliases[0], c[1]) for c in scored_candidates])

        return (
            candidate_to_canonical(top_candidate, self.kb),
            top_score,
            torch.tensor(top_vector),
        )

    def normalize_mention_vector(
        self, mention_vector: torch.Tensor, doc_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard normalization of mention vector
        - weighted combination of entity and doc vector
        - l1 normalize
        """
        vector = (
            1 - DOC_VECTOR_WEIGHT
        ) * mention_vector + DOC_VECTOR_WEIGHT * doc_vector

        norm_vector = l1_regularize(vector)
        return norm_vector

    @overrides(AbstractCandidateSelector)
    def select_candidate(
        self,
        term: str,
        mention_vector: torch.Tensor,
        doc_vector: torch.Tensor,
    ) -> EntityScoreVector | None:
        """
        Generate & select candidates for a list of mention texts
        """
        norm_vector = self.normalize_mention_vector(mention_vector, doc_vector)
        candidates = self._get_candidates(term)
        return self._get_best_canonical(norm_vector, candidates)

    def select_candidate_from_entity(
        self, entity: DocEntity
    ) -> EntityScoreVector | None:
        """
        Generate & select candidates for a list of mention texts
        """
        if not entity.spacy_doc:
            raise ValueError("Spacy doc required")

        # if entity.vector == entity.spacy_doc.vector:
        #     logger.warning("Vector and spacy doc are the same")

        norm_vector = self.normalize_mention_vector(
            torch.tensor(entity.vector), torch.tensor(entity.spacy_doc.vector)
        )
        candidates = self._get_candidates(entity.normalized_term)

        return self._get_best_canonical(norm_vector, candidates)

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        res = self.select_candidate_from_entity(entity)

        if res is None:
            return None

        return res[0]
