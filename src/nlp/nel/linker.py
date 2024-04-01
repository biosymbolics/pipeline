"""
Term Normalizer
"""

import logging
from typing import Iterable, Sequence


from nlp.ner.types import CanonicalEntity, DocEntity

from .candidate_selector import AbstractCandidateSelector, CandidateSelectorType


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CANDIDATE_SELECTOR_MODULE = "core.ner.linker"


class TermLinker:
    """
    TermLinker

    - associates UMLS canonical entity if found
    - adds synonyms to synonym store
    - if no canonical entry found, then closest synonym is used

    Example:
        >>> linker = TermLinker()
        >>> linker(["DNMT1", "DNMT1 inhibitor"])
        >>> linker(["DNMT1 protein synthase inhibitor"])
    """

    def __init__(self, candidate_selector: AbstractCandidateSelector):
        """
        Initialize term normalizer using existing model

        Use `create` to instantiate with async dependencies
        """
        self.candidate_selector = candidate_selector

    @classmethod
    async def create(
        cls,
        candidate_selector_type: CandidateSelectorType = "CandidateSelector",
        *args,
        **kwargs,
    ):
        modules = __import__(
            CANDIDATE_SELECTOR_MODULE, fromlist=[candidate_selector_type]
        )
        candidate_selector: AbstractCandidateSelector = await getattr(
            modules, candidate_selector_type
        ).create(*args, **kwargs)
        return cls(candidate_selector)

    async def link(self, entities: Iterable[DocEntity]) -> Iterable[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            entities (Sequence[DocEntity] | Iterable[DocEntity]): list of entities to link
        """

        candidates = [c async for c in self.candidate_selector(entities)]  # type: ignore
        return [
            DocEntity.merge(
                e,
                canonical_entity=c
                or CanonicalEntity.create(e.normalized_term or e.term),
            )
            for e, c in zip(iter(entities), candidates)
        ]

    async def __call__(
        self, entity_set: Sequence[DocEntity] | Iterable[DocEntity]
    ) -> Iterable[DocEntity]:
        return await self.link(entity_set)
