from abc import abstractmethod
from typing import TypeVar

from core.ner.types import DocEntity

from abc import abstractmethod
from typing import Literal, TypeVar
from scispacy.candidate_generation import MentionCandidate
import torch

from core.ner.types import CanonicalEntity, DocEntity


CandidateSelectorType = Literal[
    "CandidateSelector",
    "CompositeCandidateSelector",
]

EntityWithScore = tuple[CanonicalEntity, float]
CandidateScore = tuple[MentionCandidate, float]

EntityWithScoreVector = tuple[CanonicalEntity, float, torch.Tensor]

ST = TypeVar("ST", bound=EntityWithScore)


class AbstractCandidateSelector(object):
    """
    Base class for candidate selectors
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def select_candidate_from_entity(self, entity: DocEntity) -> ST | None:  # type: ignore # TODO
        """
        Generate & select candidates for a mention text, returning best candidate & score
        """
        raise NotImplementedError

    @abstractmethod
    async def select_candidate(self, text: str, vector: torch.Tensor | None = None) -> ST | None:  # type: ignore # TODO
        """
        Generate & select candidates for a mention text, returning best candidate & score
        """
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        raise NotImplementedError


ES = TypeVar("ES", bound=EntityWithScore)
