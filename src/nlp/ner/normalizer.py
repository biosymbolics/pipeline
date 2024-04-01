"""
Linking/cleaning of terms
"""

import time
from typing import AsyncGenerator, Iterable, Sequence
import logging


from nlp.ner.cleaning import CleanFunction, EntityCleaner
from nlp.nel import TermLinker
from nlp.nel.candidate_selector import CandidateSelectorType
from nlp.ner.spacy import get_transformer_nlp
from nlp.ner.types import DocEntity
from utils.list import batch


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TermNormalizer:
    """
    Normalizes and attempts to link to canonical entity
    If no canonical entity found, then normalized term is returned.
    """

    def __init__(
        self,
        link: bool,
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
        self.link = link
        self.nlp = get_transformer_nlp()

    @classmethod
    async def create(
        cls,
        link: bool = True,
        candidate_selector_type: CandidateSelectorType | None = None,
        *args,
        **kwargs,
    ):
        term_linker: TermLinker | None = await (
            TermLinker.create(candidate_selector_type)
            if candidate_selector_type
            else TermLinker.create()
        )

        return cls(link, term_linker, *args, **kwargs)

    async def normalize(self, doc_entities: Sequence[DocEntity]) -> Iterable[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            doc_entities (Sequence[DocEntity]): list of terms to normalize
        """
        cleaned_entities = self.cleaner.clean(doc_entities, remove_suppressed=False)

        if self.link:
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
            batch_size (int): batch size for processing terms
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
            start = time.monotonic()
            docs = list(self.nlp.pipe(b))

            doc_entities = [
                DocEntity.create(term, vector=vector, spacy_doc=doc)
                for term, doc, vector in zip(b, docs, batched_vectors[i])
            ]
            if self.link:
                linked = await self.term_linker.link(doc_entities)
                for link in linked:
                    yield link
            else:
                for entity in doc_entities:
                    yield entity

    async def __call__(self, doc_entities: Sequence[DocEntity]) -> Iterable[DocEntity]:
        return await self.normalize(doc_entities)
