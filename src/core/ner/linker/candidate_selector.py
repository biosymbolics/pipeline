from typing import Sequence
from annoy import AnnoyIndex
from scispacy.candidate_generation import CandidateGenerator, MentionCandidate
from core.ner.spacy import Spacy
import logging

from core.ner.types import CanonicalEntity, DocEntity
from constants.umls import UMLS_CUI_SUPPRESSIONS
from data.domain.biomedical.umls import clean_umls_name, get_best_umls_candidate
from utils.encoding.text_encoder import WORD_VECTOR_LENGTH

DEFAULT_K = 25
MIN_SIMILARITY = 0.85  # only used if entity.embeddings is None
MAX_DISTANCE = 0.5

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
        max_distance: float = MAX_DISTANCE,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.min_similarity = min_similarity
        self.max_distance = max_distance
        self.nlp = Spacy.get_instance(disable=["ner", "parser", "tagger"])

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
        umls_ann = AnnoyIndex(WORD_VECTOR_LENGTH, metric="angular")

        candidate_docs = self.nlp.pipe([c.aliases[0] for c in candidates])
        for i, doc in enumerate(candidate_docs):
            umls_ann.add_item(i, doc.vector)

        umls_ann.build(len(candidates))
        return umls_ann

    def get_best_by_semantic_similarity(
        self, mention_embeddings: list[float], candidates: Sequence[MentionCandidate]
    ) -> MentionCandidate | None:
        """
        Get best candidate by semantic similarity
        """
        umls_ann = self.create_ann_index(candidates)
        cui, dist = umls_ann.get_nns_by_vector(
            mention_embeddings, 10, search_k=10, include_distances=True
        )
        top_candidate = candidates[cui[0]]
        distance = dist[cui[0]]
        logger.info([(candidates[c], d) for c, d in list(zip(cui, dist))])

        if distance > self.max_distance:
            logger.warning(
                "Best candidate (%s) too distant (%s)", top_candidate, distance
            )
            return None

        return top_candidate

    def _get_best_canonical(
        self, candidates: Sequence[MentionCandidate], entity: DocEntity
    ) -> CanonicalEntity | None:
        """
        Get canonical candidate if suggestions exceed min similarity
        """

        if entity.embeddings is None:
            top_candidate = self.get_best_by_rules(candidates)
        else:
            top_candidate = self.get_best_by_semantic_similarity(
                entity.embeddings,
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

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate candidates for a list of mention texts

        If the initial top candidate isn't of sufficient similarity, generate a composite candidate.
        """
        candidates = self._get_candidates(entity.term)

        return self._get_best_canonical(candidates, entity)
