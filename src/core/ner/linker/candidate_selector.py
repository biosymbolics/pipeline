from scispacy.candidate_generation import (
    CandidateGenerator,
    MentionCandidate,
)
import logging

from core.ner.types import CanonicalEntity, DocEntity

from .types import AbstractCandidateSelector
from .utils import candidate_to_canonical, apply_umls_word_overrides

MIN_SIMILARITY = 0.85
UMLS_KB = None
K = 5  # arbitrary; maybe 1 is fine?


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CandidateSelector(AbstractCandidateSelector):
    """
    Wraps a CandidateGenerator to select the best candidate for a mention
    Returns the candidate with the highest similarity score
    """

    def __init__(self, *args, min_similarity: float = MIN_SIMILARITY, **kwargs):
        global UMLS_KB

        if UMLS_KB is None:
            from scispacy.linking_utils import UmlsKnowledgeBase

            UMLS_KB = UmlsKnowledgeBase()

        self.kb = UMLS_KB
        self.candidate_generator = CandidateGenerator(*args, kb=UMLS_KB, **kwargs)

        if min_similarity > 1:
            raise ValueError("min_similarity must be <= 1")
        elif min_similarity > 0.85:
            logger.warning(
                "min_similarity is high, this may result in many missed matches"
            )

        self.min_similarity = min_similarity

    def _get_best_candidate(self, text: str) -> MentionCandidate | None:
        """
        Wrapper around super().__call__ that handles word overrides
        """
        candidates = self.candidate_generator([text], k=K)[0]
        if len(candidates) == 0:
            logger.warning(f"No candidates found for {text}")
            return None

        candidate = candidates[0]

        if candidate.similarities[0] < self.min_similarity:
            logger.warning(
                f"Best candidate similarity {candidate.similarities[0]} is below threshold {self.min_similarity}"
            )
            return None

        with_overrides = apply_umls_word_overrides(text, [candidate])
        return with_overrides[0]

    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        candidate = self._get_best_candidate(entity.term)

        if candidate is None:
            return None

        return candidate_to_canonical(candidate, self.kb)
