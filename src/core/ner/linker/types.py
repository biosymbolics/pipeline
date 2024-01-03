from abc import abstractmethod
from typing import Literal

from core.ner.types import CanonicalEntity, DocEntity


CandidateSelectorType = Literal[
    "SemanticCandidateSelector",
    "CompositeCandidateSelector",
    "CandidateSelector",
]


class AbstractCandidateSelector(object):
    """
    Base class for candidate selectors
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, entity: DocEntity) -> CanonicalEntity | None:
        """
        Generate & select candidates for a list of mention texts
        """
        raise NotImplementedError
