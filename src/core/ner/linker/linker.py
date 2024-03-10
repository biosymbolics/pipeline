"""
Term Normalizer
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import AsyncIterable, Iterable, Sequence

from utils.async_utils import gather_with_concurrency_limit
from utils.list import batch

from .candidate_selector import AbstractCandidateSelector, CandidateSelectorType

from ..types import CanonicalEntity, DocEntity

LinkedEntityMap = dict[str, CanonicalEntity]

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
        **kwargs
    ):
        modules = __import__(
            CANDIDATE_SELECTOR_MODULE, fromlist=[candidate_selector_type]
        )
        candidate_selector: AbstractCandidateSelector = await getattr(
            modules, candidate_selector_type
        ).create(*args, **kwargs)
        return cls(candidate_selector)

    async def link(
        self, entities: Sequence[DocEntity] | Iterable[DocEntity]
    ) -> AsyncIterable[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            entities (Sequence[DocEntity] | Iterable[DocEntity]): list of entities to link
        """
        # generate the candidates (kinda slow)
        for e in entities:
            ce = await self.candidate_selector(e)
            yield DocEntity.merge(
                e,
                canonical_entity=ce
                or CanonicalEntity(
                    id="", name=e.normalized_term or e.term, aliases=[e.term]
                ),
            )

    async def __call__(
        self, entity_set: Sequence[DocEntity] | Iterable[DocEntity]
    ) -> AsyncIterable[DocEntity]:
        return self.link(entity_set)
