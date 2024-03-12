"""
Linking/cleaning of terms
"""

import time
from typing import AsyncGenerator, Iterable, Sequence
import logging


from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker.linker import TermLinker
from core.ner.spacy import get_transformer_nlp
from core.ner.types import DocEntity
from utils.list import batch

from .linker.candidate_selector import CandidateSelectorType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TermNormalizer:
    """
    Normalizes and attempts to link entities.
    If no canonical entity found, then normalized term is returned.
    """

    def __init__(
        self,
        term_linker: TermLinker,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        """
        Initialize term normalizer

        Use `create` to instantiate with async dependencies
        """
        self.term_linker = term_linker
        self.cleaner = EntityCleaner(
            additional_cleaners=additional_cleaners,
        )
        self.nlp = get_transformer_nlp()

    @classmethod
    async def create(
        cls,
        candidate_selector_type: CandidateSelectorType | None = None,
        *args,
        **kwargs,
    ):
        term_linker: TermLinker | None = await (
            TermLinker.create(candidate_selector_type)
            if candidate_selector_type
            else TermLinker.create()
        )

        return cls(term_linker, *args, **kwargs)

    async def normalize(self, doc_entities: Sequence[DocEntity]) -> Iterable[DocEntity]:
        """
        Normalize and link terms to canonical entities
        """
        # removed_suppressed must be false to properly index against original terms
        cleaned_entities = self.cleaner.clean(doc_entities, remove_suppressed=False)

        if self.term_linker is not None:
            return await self.term_linker.link(cleaned_entities)

        return cleaned_entities

    async def normalize_strings(
        self,
        terms: Sequence[str],
        vectors: Sequence[list[float]] | None = None,
        batch_size: int = 10000,
    ) -> AsyncGenerator[DocEntity, None]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize
            vectors (Sequence[list[float]]): list of vectors for each term - optional.
        """
        start = time.monotonic()
        logger.info("Normalizing %s terms", len(terms))

        if vectors is not None and len(terms) != len(vectors):
            raise ValueError("terms and vectors must be the same length")

        clean_terms = self.cleaner.clean(terms, remove_suppressed=False)
        batched_terms = batch(clean_terms, batch_size)
        batched_vectors = batch(vectors or [None for _ in terms], batch_size)  # type: ignore

        for i, b in enumerate(batched_terms):
            logger.info("Batch %s (last took: %ss)", i, round(time.monotonic() - start))
            docs = list(self.nlp.pipe(b))

            doc_entities = [
                DocEntity.create(term, vector=vector, spacy_doc=doc)
                for term, doc, vector in zip(b, docs, batched_vectors[i])
            ]

            linked = await self.term_linker.link(doc_entities)
            for link in linked:
                yield link

    async def __call__(self, doc_entities: Sequence[DocEntity]) -> Iterable[DocEntity]:
        return await self.normalize(doc_entities)
