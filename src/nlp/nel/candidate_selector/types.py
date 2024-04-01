from abc import abstractmethod
from typing import AsyncIterable, AsyncIterator, Iterable, Sequence, TypeVar

from nlp.ner.types import DocEntity

from abc import abstractmethod
from typing import Literal, TypeVar
from scispacy.candidate_generation import MentionCandidate
import torch

from nlp.ner.types import CanonicalEntity, DocEntity


CandidateSelectorType = Literal[
    "CandidateSelector",
    "CompositeCandidateSelector",
]
EntityWithScore = tuple[CanonicalEntity, float]
CandidateScore = tuple[MentionCandidate, float]
EntityWithScoreVector = tuple[CanonicalEntity, float, torch.Tensor]


class AbstractCandidateSelector(object):
    """
    Base class for candidate selectors
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def select_candidates_from_entities(
        self, entities: Iterable[DocEntity]
    ) -> AsyncIterable[EntityWithScore | None]:
        """
        Generate & select candidates for a mention text, returning best candidate & score
        """
        raise NotImplementedError

    @abstractmethod
    def select_candidates(
        self,
        texts: list[str],
        vectors: list[torch.Tensor] | None = None,
        min_similarity: float = 0.85,
        is_composite: bool = False,
    ) -> AsyncIterator[EntityWithScore | None]:
        """
        Generate & select candidates for a mention text, returning best candidate & score
        """
        raise NotImplementedError

    @abstractmethod
    async def __call__(
        self, entities: Iterable[DocEntity]
    ) -> AsyncIterable[CanonicalEntity | None]:
        """
        Generate & select candidates for a list of mention texts
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    async def create(cls, *args, **kwargs):
        raise NotImplementedError


ES = TypeVar("ES", bound=EntityWithScore)
