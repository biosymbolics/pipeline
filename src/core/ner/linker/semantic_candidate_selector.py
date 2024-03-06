import time
from typing import Sequence
from annoy import AnnoyIndex
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
import torch
import logging

from core.ner.spacy import get_transformer_nlp
from core.ner.types import CanonicalEntity, DocEntity
from core.vector.vectorizer import Vectorizer
from utils.classes import overrides
from utils.tensor import l1_regularize

from .types import AbstractCandidateSelector, EntityWithScoreVector
from .utils import (
    apply_umls_word_overrides,
    candidate_to_canonical,
    score_semantic_candidate,
)

DEFAULT_K = 20
MIN_SIMILARITY = 1.0
UMLS_KB = None
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
        **kwargs,
    ):
        global UMLS_KB

        if UMLS_KB is None:
            from scispacy.linking_utils import UmlsKnowledgeBase

            UMLS_KB = UmlsKnowledgeBase()

        self.kb = UMLS_KB
        self.candidate_generator = CandidateGenerator(*args, kb=UMLS_KB, **kwargs)
        self.min_similarity = min_similarity
        self.vector_length = vector_length

        # must use the same model!
        self.vectorizer = Vectorizer()
        self.nlp = get_transformer_nlp()

        self.vector_cache: dict[str, torch.Tensor] = {}

    def _batch_vectorize(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """
        Vectorize texts
        """
        new_vecs = self.vectorizer.vectorize(
            [t for t in texts if t not in self.vector_cache.keys()]
        )
        cached_vecs = [
            self.vector_cache[t] for t in texts if t in self.vector_cache.keys()
        ]
        reg_new_vecs = [l1_regularize(vec) for vec in new_vecs]
        self.vector_cache.update({text: vec for text, vec in zip(texts, reg_new_vecs)})
        return reg_new_vecs + cached_vecs

    def _get_candidates(self, text: str) -> list[MentionCandidate]:
        """
        Wrapper around candidate generator call that handles word overrides
        """
        candidates = self.candidate_generator([text], k=DEFAULT_K)[0]
        with_overrides = apply_umls_word_overrides(text, candidates)
        return with_overrides

    def _score_candidate(
        self,
        concept_id: str,
        matching_aliases: Sequence[str],
        mention_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        syntactic_similarity: float,
        is_composite: bool,
    ) -> float:
        return score_semantic_candidate(
            concept_id,
            self.kb.cui_to_entity[concept_id].canonical_name,
            self.kb.cui_to_entity[concept_id].types,
            self.kb.cui_to_entity[concept_id].aliases,
            matching_aliases=matching_aliases,
            original_vector=mention_vector,
            candidate_vector=candidate_vector,
            syntactic_similarity=syntactic_similarity,
            is_composite=is_composite,
        )

    def create_ann_index(
        self,
        candidates: Sequence[MentionCandidate],
    ) -> AnnoyIndex:
        """
        Create an Annoy index for a list of candidates

        Metrics other than 'angular' work horribly
        """
        start = time.monotonic()
        umls_ann = AnnoyIndex(self.vector_length, metric="angular")

        canonical_names = [
            self.kb.cui_to_entity[c.concept_id].canonical_name.lower()
            for c in candidates
        ]

        vectors = self._batch_vectorize(canonical_names)

        for i in range(len(candidates)):
            combined_vec = vectors[i]
            umls_ann.add_item(i, combined_vec.detach().cpu().numpy())

        umls_ann.build(len(candidates))
        logger.debug(
            "Took %ss to vectorize & build Annoy index (%s)",
            time.monotonic() - start,
            len(candidates),
        )
        return umls_ann

    def _get_best_canonical(
        self,
        mention_vector: torch.Tensor,
        candidates: Sequence[MentionCandidate],
        is_composite: bool,
        n: int = round(DEFAULT_K / 2),
    ) -> EntityWithScoreVector | None:
        """
        Get best candidate by semantic similarity
        """
        if len(candidates) == 0:
            logger.warning("get_best_canonical called with no candidates")
            return None

        if len(mention_vector) == 0:
            logger.warning(
                "No vector for %s, probably OOD (%s)",
                candidates[0].aliases[0],
                mention_vector,
            )
            return None

        umls_ann = self.create_ann_index(candidates)
        norm_vector = l1_regularize(mention_vector)
        ids = umls_ann.get_nns_by_vector(norm_vector.tolist(), n, search_k=-1)

        if len(ids) == 0:
            logger.warning("No candidates for %s", candidates[0].aliases[0])
            return None

        scored_candidates = sorted(
            [
                (
                    candidates[id],
                    self._score_candidate(
                        candidates[id].concept_id,
                        candidates[id].aliases,
                        norm_vector,
                        candidate_vector=torch.tensor(umls_ann.get_item_vector(id)),
                        syntactic_similarity=candidates[id].similarities[0],
                        is_composite=is_composite,
                    ),
                    umls_ann.get_item_vector(id),
                )
                for id in ids
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        logger.debug("Scored candidates: %s", [sc[0] for sc in scored_candidates])
        top_candidate = scored_candidates[0][0]
        top_score = scored_candidates[0][1]
        top_vector = scored_candidates[0][2]

        return (
            candidate_to_canonical(top_candidate, self.kb),
            top_score,
            torch.tensor(top_vector),
        )

    @overrides(AbstractCandidateSelector)
    def select_candidate(
        self,
        term: str,
        mention_vector: torch.Tensor,
        is_composite: bool,
    ) -> EntityWithScoreVector | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidates = self._get_candidates(term)

        return self._get_best_canonical(
            mention_vector, candidates, is_composite=is_composite
        )

    @overrides(AbstractCandidateSelector)
    def select_candidate_from_entity(
        self, entity: DocEntity, is_composite: bool
    ) -> EntityWithScoreVector | None:
        """
        Generate & select candidates for a list of mention texts

        Args:
            entity (DocEntity): entity to link
        """
        if entity.vector is None:
            raise ValueError("Vector required")

        return self.select_candidate(
            entity.normalized_term,
            entity.vector,
            is_composite=is_composite,
        )

    def __call__(
        self, entity: DocEntity, is_composite: bool = False
    ) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        res = self.select_candidate_from_entity(entity, is_composite)

        if res is None:
            return None

        return res[0]
