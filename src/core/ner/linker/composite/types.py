from abc import abstractmethod
from typing import TypeVar

from core.ner.types import DocEntity

from ..types import EntityWithScore

ES = TypeVar("ES", bound=EntityWithScore)


class AbstractCompositeCandidateSelector(object):
    """
    Base class for composite candidate selectors
    """

    @abstractmethod
    def generate_candidate(self, entity: DocEntity) -> ES | None:
        """
        Generate a composite candidate for a mention text
        """
        raise NotImplementedError
