"""
Linking/cleaning of terms
"""

from typing import AsyncIterable, Iterable, Sequence
import logging


from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker.linker import TermLinker
from core.ner.spacy import get_transformer_nlp
from core.ner.types import DocEntity

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
        term_linker: TermLinker | None = None,
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
        link: bool = True,
        candidate_selector_type: CandidateSelectorType | None = None,
        *args,
        **kwargs
    ):
        if link:
            term_linker: TermLinker | None = await (
                TermLinker.create(candidate_selector_type)
                if candidate_selector_type
                else TermLinker.create()
            )
        else:
            term_linker = None

        return cls(term_linker, *args, **kwargs)

    async def normalize(
        self, doc_entities: Sequence[DocEntity]
    ) -> AsyncIterable[DocEntity] | list[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
        """
        # removed_suppressed must be false to properly index against original terms
        cleaned_entities = self.cleaner.clean(doc_entities, remove_suppressed=False)

        if self.term_linker is not None:
            return self.term_linker.link(cleaned_entities)

        return cleaned_entities

    async def normalize_strings(
        self, terms: Sequence[str], vectors: Sequence[list[float]] | None = None
    ) -> AsyncIterable[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize
            vectors (Sequence[list[float]]): list of vectors for each term - optional.
        """

        logger.info("Normalizing %s terms", len(terms))

        if vectors is not None and len(terms) != len(vectors):
            raise ValueError("terms and vectors must be the same length")

        def clean_and_docify() -> Iterable[DocEntity]:
            clean_terms = self.cleaner.clean(terms, remove_suppressed=False)
            docs = self.nlp.pipe(clean_terms)
            for term, vector, doc in zip(
                clean_terms, vectors or [None for _ in terms], docs
            ):
                yield DocEntity.create(
                    term=term,
                    vector=vector,
                    spacy_doc=doc,
                )

        if self.term_linker is not None:
            return self.term_linker.link(clean_and_docify())

        raise ValueError("TermLinker is not defined")

    async def __call__(
        self, doc_entities: Sequence[DocEntity]
    ) -> AsyncIterable[DocEntity] | list[DocEntity]:
        return await self.normalize(doc_entities)
