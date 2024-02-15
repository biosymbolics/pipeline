import time
from typing import Sequence
from annoy import AnnoyIndex
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
import torch
import logging

from core.ner.spacy import get_transformer_nlp
from core.ner.types import CanonicalEntity, DocEntity
from constants.umls import PREFERRED_UMLS_TYPES
from utils.classes import overrides

from .types import AbstractCandidateSelector, EntityWithScoreVector
from .utils import (
    apply_umls_word_overrides,
    candidate_to_canonical,
    l1_regularize,
    score_semantic_candidate,
)

DEFAULT_K = 20
MIN_SIMILARITY = 1.2
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

    def _batch_vectorize(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """
        Vectorize texts
        """
        docs = list(self.nlp.pipe(texts))
        bert_vecs = [
            (
                l1_regularize(torch.tensor(doc.vector))
                if len(doc.vector) > 0
                else torch.zeros(self.vector_length)
            )
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
        # metrics other than 'angular' work horribly
        umls_ann = AnnoyIndex(self.vector_length, metric="angular")

        canonical_names = [
            self.kb.cui_to_entity[c.concept_id].canonical_name.lower()
            for c in candidates
        ]
        # only use the first alias, since that's the most syntactically similar to the mention
        tuis = [self.kb.cui_to_entity[c.concept_id].types[0] for c in candidates]

        # get (preferred) UMLS types these to use in semantic similarity determination
        # because otherwise terms are totally without disambiguating context
        types = [PREFERRED_UMLS_TYPES.get(tui) or tui for tui in tuis]

        vectors = self._batch_vectorize(canonical_names + types)
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
        matching_aliases: Sequence[str],
        mention_vector: torch.Tensor,
        candidate_vector: torch.Tensor,
        syntactic_similarity: float,
        semantic_distance: float,
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
            semantic_distance=semantic_distance,
            is_composite=is_composite,
        )

    def _get_best_canonical(
        self,
        mention_vector: torch.Tensor,
        candidates: Sequence[MentionCandidate],
        is_composite: bool,
    ) -> EntityWithScoreVector | None:
        """
        Get best candidate by semantic similarity
        """
        if len(candidates) == 0:
            logger.warning("get_best_canoical called with no candidates")
            return None

        if len(mention_vector) == 0:
            logger.warning(
                "No vector for %s, probably OOD (%s)",
                candidates[0].aliases[0],
                mention_vector,
            )
            return None

        norm_vector = l1_regularize(mention_vector)
        umls_ann = self.create_ann_index(candidates)

        ids, distances = umls_ann.get_nns_by_vector(
            norm_vector.tolist(), 10, search_k=-1, include_distances=True
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
                        candidates[id].aliases,
                        norm_vector,
                        candidate_vector=torch.tensor(umls_ann.get_item_vector(id)),
                        syntactic_similarity=candidates[id].similarities[0],
                        semantic_distance=distances[i],
                        is_composite=is_composite,
                    ),
                    umls_ann.get_item_vector(id),
                )
                for i, id in enumerate(ids)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("Sorted candidates: %s", [sc[0] for sc in scored_candidates])
        top_candidate = scored_candidates[0][0]
        top_score = scored_candidates[0][1]
        top_vector = scored_candidates[0][2]

        print([(c[0].aliases[0], c[1]) for c in scored_candidates])

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
        if not entity.vector:
            raise ValueError("Vector required")

        return self.select_candidate(
            entity.normalized_term,
            torch.tensor(entity.vector),
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
