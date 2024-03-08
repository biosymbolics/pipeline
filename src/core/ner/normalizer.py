"""
Linking/cleaning of terms
"""

import time
from typing import Sequence
import logging
from spacy.tokens import Doc
import torch


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

    Usage:
        normalizer = TermNormalizer()
        terms = normalizer.normalize([
            "Tipranavir (TPV)",
            "BILR 355 - D4",
            "bivatuzumab mertansine",
            "BIBT 986 BS - single rising dose",
            "RDEA3170 10 mg",
        ])
        [(t[0], t[1].name) for t in terms]
    """

    def __init__(
        self,
        link: bool = True,
        candidate_selector: CandidateSelectorType | None = None,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        """
        Initialize term normalizer
        """
        if link:
            self.term_linker: TermLinker | None = (
                TermLinker(candidate_selector) if candidate_selector else TermLinker()
            )
        else:
            self.term_linker = None

        self.cleaner = EntityCleaner(
            additional_cleaners=additional_cleaners,
        )

        self.nlp = get_transformer_nlp()

    async def normalize(self, doc_entities: Sequence[DocEntity]) -> list[DocEntity]:
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
            return await self.term_linker.link(cleaned_entities)

        return cleaned_entities

    async def normalize_strings(
        self, terms: Sequence[str], vectors: Sequence[Sequence[float]] | None = None
    ) -> list[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize
            vectors (Sequence[Sequence[float]]): list of vectors for each term - optional.
        """

        logger.info("Normalizing %s terms", len(terms))

        if vectors is not None and len(terms) != len(vectors):
            raise ValueError("terms and vectors must be the same length")

        def get_vecs(
            vectors,
        ) -> tuple[list[Doc] | list[None], list[torch.Tensor] | list[None]]:
            # if no vectors AND candidate selectors are semantic, generate docs / vectors
            if not vectors:
                if self.nlp is None:
                    raise ValueError(
                        "Vectorizer required for semantic candidate selector"
                    )

                start = time.monotonic()
                docs = list(self.nlp.pipe(clean_terms))
                logger.info(
                    "Took %ss to docify %s terms",
                    round(time.monotonic() - start),
                    len(clean_terms),
                )
                return docs, vectors or [torch.tensor(doc.vector) for doc in docs]

            return [None for _ in terms], vectors or [None for _ in terms]

        clean_terms = self.cleaner.clean(terms, remove_suppressed=False)
        docs, _vectors = get_vecs(vectors)

        doc_ents = [
            DocEntity.create(
                term=term,
                vector=torch.tensor(vector),
                spacy_doc=doc,  # doc for term, NOT the source doc (confusing!!)
            )
            for term, vector, doc in zip(clean_terms, _vectors, docs)
        ]

        if self.term_linker is not None:
            return await self.term_linker.link(doc_ents)

        return doc_ents

    async def __call__(self, doc_entities: Sequence[DocEntity]) -> list[DocEntity]:
        return await self.normalize(doc_entities)
