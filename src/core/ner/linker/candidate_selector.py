from functools import reduce
import time
from typing import Sequence
from annoy import AnnoyIndex
import joblib
from numpy import mean
from pydash import flatten
from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
    cached_path,
)
from spacy.tokens import Doc
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import logging

from core.ner.spacy import get_transformer_nlp
from core.ner.types import CanonicalEntity
from constants.umls import UMLS_CUI_SUPPRESSIONS
from data.domain.biomedical.umls import clean_umls_name, get_best_umls_candidate

DEFAULT_K = 25
MIN_SIMILARITY = 1.21
UMLS_KB = None
BIOSYM_UMLS_TFIDF_PATH = (
    "https://biosym-umls-tfidf.s3.amazonaws.com/tfidf_vectorizer.joblib"
)

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
            torch.tensor(doc._.trf_data.tensors[0]).squeeze(0).mean(dim=0).squeeze()
            for doc in docs
        ]
        tfidf_vecs = torch.tensor(self.tfidf.transform(texts).toarray())
        projected_tfidf = torch.tensor_split(
            self.tfidf_ll.forward(tfidf_vecs), len(texts)
        )
        return [
            bert_vec + tfidf_vec.squeeze()
            for bert_vec, tfidf_vec in zip(bert_vecs, projected_tfidf)
        ]

    def vectorize_doc(self, doc: Doc) -> tuple[torch.Tensor, Doc]:
        """
        Vectorize a text
        """
        bert_vec = torch.tensor(doc.vector)
        tfidf_vec = self.tfidf_ll.forward(
            torch.tensor(self.tfidf.transform([doc.text]).toarray()[0])
        )
        return torch.cat([bert_vec, tfidf_vec]), doc

    def vectorize(self, text: str) -> tuple[torch.Tensor, Doc]:
        """
        Vectorize a text
        """
        doc = self.nlp(text)
        return self.vectorize_doc(doc)

    def get_best_by_rules(
        self, candidates: Sequence[MentionCandidate]
    ) -> MentionCandidate | None:
        """
        Wrapper for get_best_umls_candidate
        (legacy-ish, since semantic similarity is better)
        """

        return get_best_umls_candidate(
            candidates,
            self.min_similarity,
            self.kb,
            list(CANDIDATE_CUI_SUPPRESSIONS.keys()),
        )

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
        aliases = [c.aliases[0].lower() for c in candidates]

        vectors = self.batch_vectorize(canonical_names + aliases)
        cn_vectors = vectors[0 : len(canonical_names)]
        alias_vectors = vectors[len(canonical_names) :]

        for i in range(len(candidates)):
            combined_vec = 0.6 * cn_vectors[i] + 0.4 * alias_vectors[i]
            umls_ann.add_item(i, combined_vec.detach().cpu().numpy())

        umls_ann.build(len(candidates))
        logger.debug("Took %s seconds to build Annoy index", time.monotonic() - start)
        return umls_ann

    def get_best_by_semantic_similarity(
        self, embeddings: list[float], candidates: Sequence[MentionCandidate]
    ) -> MentionCandidate | None:
        """
        Get best candidate by semantic similarity
        """
        if mean(embeddings) == 0:
            logger.warning("No embeddings for %s", candidates[0].aliases[0])
            return None

        umls_ann = self.create_ann_index(candidates)
        indexes, dist = umls_ann.get_nns_by_vector(
            embeddings, 10, search_k=-1, include_distances=True
        )

        if len(indexes) == 0:
            return None

        mixed_score_candidates = sorted(
            [
                (
                    candidates[i],
                    (2 - dist[indexes.index(i)]) + (candidates[i].similarities[0] / 2),
                )
                for i in indexes
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        top_candidate = mixed_score_candidates[0][0]
        top_score = mixed_score_candidates[0][1]

        print(mixed_score_candidates)

        if top_score < self.min_similarity:
            logger.warning(
                "Best candidate (%s) too distant (%s)",
                top_candidate,
                top_score,
            )
            return None

        return top_candidate

    def _get_best_canonical(
        self,
        candidates: Sequence[MentionCandidate],
        mention_embeddings: list[float] | None = None,
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity

        Args:
            candidates (Sequence[MentionCandidate]): list of candidates
            mention_embeddings (list[float], Optional): embeddings for mention text
        """

        if mention_embeddings is None:
            top_candidate = self.get_best_by_rules(candidates)
        else:
            top_candidate = self.get_best_by_semantic_similarity(
                mention_embeddings,
                candidates,
            )

        if top_candidate is None:
            return None

        return self._candidate_to_canonical(top_candidate)

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
        self, term: str, embeddings: list[float] | None = None
    ) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidates = self._get_candidates(term)

        return self._get_best_canonical(candidates, embeddings)

    def __call__(
        self, term: str, embeddings: list[float] | None = None
    ) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        return self.select_candidate(term, embeddings)
