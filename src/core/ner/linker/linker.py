"""
Term Normalizer
"""

import asyncio
import logging
import time
from typing import Sequence

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

    async def link(self, entities: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Link term to canonical entity or synonym

        Args:
            terms (Sequence[str]): list of terms to normalize
        """
        if len(entities) == 0:
            logging.warning("No entities to link")
            return []

        start = time.monotonic()

        # generate the candidates (kinda slow)
        canonical_entities = await asyncio.gather(
            *[asyncio.create_task(self.candidate_selector(e)) for e in entities]
        )

        logging.info(
            "Completed candidate generation, took %ss (%s)",
            round(time.monotonic() - start),
            [e.term for e in entities],
        )

        linked_doc_ents = [
            DocEntity.merge(
                e,
                canonical_entity=ce
                or CanonicalEntity(
                    id="", name=e.normalized_term or e.term, aliases=[e.term]
                ),
            )
            for e, ce in zip(entities, canonical_entities)
        ]

        logging.info(
            "Completed linking batch of %s entities, took %ss",
            len(entities),
            round(time.monotonic() - start),
        )

        return linked_doc_ents

    async def __call__(self, entity_set: Sequence[DocEntity]) -> list[DocEntity]:
        return await self.link(entity_set)
