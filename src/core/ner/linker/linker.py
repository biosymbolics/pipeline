"""
Term Normalizer
"""

import logging
import time
from typing import AsyncIterable, Iterable, Sequence
from aiostream import stream

from utils.async_utils import sync_to_async

from .candidate_selector import AbstractCandidateSelector, CandidateSelectorType

from ..types import CanonicalEntity, DocEntity


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

    async def link(self, entities: Iterable[DocEntity]) -> AsyncIterable[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            entities (Sequence[DocEntity] | Iterable[DocEntity]): list of entities to link
        """
        i = 0
        start = time.monotonic()
        des = sync_to_async(entities)
        candidates = self.candidate_selector(entities)
        async for c, e in stream.zip(candidates, des):
            if not isinstance(e, DocEntity) or (
                c is not None and not isinstance(c, CanonicalEntity)
            ):
                raise ValueError(f"Expected tuple `DocEntity` and `CanonicalEntity`")

            i += 1
            if i % 500 == 0:
                logger.info("Linked %s entities in %ss", i, time.monotonic() - start)
                start = time.monotonic()
            yield DocEntity.merge(
                e,
                canonical_entity=c
                or CanonicalEntity(
                    id="", name=e.normalized_term or e.term, aliases=[e.term]
                ),
            )

    async def __call__(
        self, entity_set: Sequence[DocEntity] | Iterable[DocEntity]
    ) -> AsyncIterable[DocEntity]:
        return self.link(entity_set)
