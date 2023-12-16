import time
from typing import Sequence
from annoy import AnnoyIndex
import joblib
from numpy import mean
from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
    cached_path,
)
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import logging
from core.ner.linker.utils import similarity_with_residual_penalty, l1_normalize

from core.ner.spacy import get_transformer_nlp
from core.ner.types import CanonicalEntity
from constants.umls import (
    BIOSYM_UMLS_TFIDF_PATH,
    MOST_PREFERRED_UMLS_TYPES,
    PREFERRED_UMLS_TYPES,
    PREFERRED_UMLS_TYPES,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
)
from data.domain.biomedical.umls import clean_umls_name
from utils.list import has_intersection

DEFAULT_K = 25
MIN_SIMILARITY = 1.2
UMLS_KB = None


WORD_EMBEDDING_LENGTH = 768

CANDIDATE_CUI_SUPPRESSIONS = {
    **UMLS_CUI_SUPPRESSIONS,
    "C0432616": "Blood group antibody A",  # matches "anti", sigh
    "C1704653": "cell device",  # matches "cell"
    "C0231491": "antagonist muscle action",  # blocks better match (C4721408)
    "C0205263": "Induce (action)",
    "C1709060": "Modulator device",
    "C0179302": "Binder device",
    "C0280041": "Substituted Urea",  # matches all "substituted" terms, sigh
    "C1179435": "Protein Component",  # sigh... matches "component"
    "C0870814": "like",
    "C0080151": "Simian Acquired Immunodeficiency Syndrome",  # matches "said"
    "C0163712": "Relate - vinyl resin",
    "C2827757": "Antimicrobial Resistance Result",  # ("result") ugh
    "C1882953": "ring",
    "C0457385": "seconds",  # s
    "C0179636": "cart",  # car-t
    "C0039552": "terminally ill",
    "C0175816": "https://uts.nlm.nih.gov/uts/umls/concept/C0175816",
    "C0243072": "derivative",
    "C1744692": "NOS inhibitor",  # matches "inhibitor"
}


# map term to specified cui
COMPOSITE_WORD_OVERRIDES = {
    "modulator": "C0005525",  # "Biological Response Modifiers"
    "modulators": "C0005525",
    "binder": "C1145667",  # "Binding action"
    "binders": "C1145667",
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateSelector(CandidateGenerator, object):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
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

        super().__init__(*args, kb=UMLS_KB, **kwargs)
        self.min_similarity = min_similarity
        self.vector_length = vector_length
        self.nlp = get_transformer_nlp()
        self.tfidf: TfidfVectorizer = joblib.load(cached_path(BIOSYM_UMLS_TFIDF_PATH))
        self.tfidf_ll = torch.nn.Linear(len(self.tfidf.vocabulary_), self.vector_length)

    def batch_vectorize(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """
        Vectorize a text
        """
        docs = list(self.nlp.pipe(texts))
        bert_vecs = [
            l1_normalize(torch.tensor(doc.vector))
            if len(doc.vector) > 0
            else torch.zeros(self.vector_length)
            for doc in docs
        ]

        # tfidf_vecs = torch.tensor(self.tfidf.transform(texts).toarray())
        # projected_tfidf = torch.tensor_split(
        #     self.tfidf_ll.forward(tfidf_vecs), len(texts)
        # )

        # return [
        #     bert_vec * 0.6 + tfidf_vec.squeeze() * 0.4
        #     for bert_vec, tfidf_vec in zip(bert_vecs, projected_tfidf)
        # ]
        return bert_vecs

    @classmethod
    def _apply_word_overrides(
        cls, text: str, candidates: list[MentionCandidate]
    ) -> list[MentionCandidate]:
        """
        Certain words we match to an explicit cui (e.g. "modulator" -> "C0005525")
        """
        # look for any overrides (terms -> candidate)
        has_override = text.lower() in COMPOSITE_WORD_OVERRIDES
        if has_override:
            return [
                MentionCandidate(
                    concept_id=COMPOSITE_WORD_OVERRIDES[text.lower()],
                    aliases=[text],
                    similarities=[1],
                )
            ]
        return candidates

    def _get_candidates(self, text: str) -> list[MentionCandidate]:
        """
        Wrapper around super().__call__ that handles word overrides
        """
        candidates = super().__call__([text], k=DEFAULT_K)[0]
        with_overrides = self._apply_word_overrides(text, candidates)
        return with_overrides

    def create_ann_index(self, candidates: Sequence[MentionCandidate]) -> AnnoyIndex:
        """
        Create an Annoy index for a list of candidates
        """
        start = time.monotonic()
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
            combined_vec = (0.9 * cn_vectors[i]) + (0.1 * type_vectors[i])
            umls_ann.add_item(i, combined_vec.detach().cpu().numpy())

        umls_ann.build(len(candidates))
        logger.debug("Took %s seconds to build Annoy index", time.monotonic() - start)
        return umls_ann

    def calc_candidate_score(
        self,
        candidate: MentionCandidate,
        original_vector: list[float],
        matched_vector: list[float],
        distance: float,
    ) -> float:
        """
        Generate a score for a candidate
        """

        if candidate.concept_id in CANDIDATE_CUI_SUPPRESSIONS:
            return 0.0

        if has_intersection(
            UMLS_NAME_SUPPRESSIONS,
            self.kb.cui_to_entity[candidate.concept_id].canonical_name.split(" "),
        ):
            return 0.0

        types = self.kb.cui_to_entity[candidate.concept_id].types
        is_preferred_type = has_intersection(types, list(PREFERRED_UMLS_TYPES.keys()))
        is_most_preferred_type = has_intersection(
            types, list(MOST_PREFERRED_UMLS_TYPES.keys())
        )
        type_score = 0.75 if not is_preferred_type else 1.1**is_most_preferred_type
        semantic_similarity = similarity_with_residual_penalty(
            torch.tensor(original_vector), torch.tensor(matched_vector), distance
        )
        syntactic_similarity = candidate.similarities[0]
        return (semantic_similarity + syntactic_similarity) * type_score

    def _get_best_canonical(
        self, vector: list[float], candidates: Sequence[MentionCandidate]
    ) -> tuple[CanonicalEntity | None, float, list[float]]:
        """
        Get best candidate by semantic similarity
        """
        if len(vector) == 0:
            logger.warning(
                "No vector for %s, probably OOD (%s)",
                candidates[0].aliases[0],
                vector,
            )
            return (None, 0.0, [])

        umls_ann = self.create_ann_index(candidates)

        ids, distances = umls_ann.get_nns_by_vector(
            vector, 10, search_k=-1, include_distances=True
        )

        if len(ids) == 0:
            logger.warning("No candidates for %s", candidates[0].aliases[0])
            return (None, 0.0, [])

        mixed_score_candidates = sorted(
            [
                (
                    candidates[id],
                    self.calc_candidate_score(
                        candidates[id],
                        vector,
                        matched_vector=umls_ann.get_item_vector(id),
                        distance=distances[i],
                    ),
                    umls_ann.get_item_vector(id),
                )
                for i, id in enumerate(ids)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidate = mixed_score_candidates[0][0]
        top_score = mixed_score_candidates[0][1]
        top_vector = mixed_score_candidates[0][2]

        print([(c[0].aliases[0], c[1]) for c in mixed_score_candidates])

        return (self._candidate_to_canonical(top_candidate), top_score, top_vector)

    def _candidate_to_canonical(self, candidate: MentionCandidate) -> CanonicalEntity:
        """
        Convert a MentionCandidate to a CanonicalEntity
        """
        # go to kb to get canonical name
        entity = self.kb.cui_to_entity[candidate.concept_id]
        name = clean_umls_name(
            entity.concept_id,
            entity.canonical_name,
            entity.aliases,
            entity.types,
            False,
        )

        return CanonicalEntity(
            id=entity.concept_id,
            ids=[entity.concept_id],
            name=name,
            aliases=entity.aliases,
            description=entity.definition,
            types=entity.types,
        )

    def select_candidate(
        self, term: str, vector: list[float]
    ) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidates = self._get_candidates(term)
        best_canonical, _, _ = self._get_best_canonical(vector, candidates)
        return best_canonical

    def __call__(self, term: str, vector: list[float]) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        return self.select_candidate(term, vector)
