"""
Linking/cleaning of terms
"""
from typing import Sequence

from pydash import flatten
from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker.linker import TermLinker
from core.ner.types import CanonicalEntity, DocEntity


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
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        """
        Initialize term normalizer
        """
        if link:
            self.term_linker: TermLinker | None = TermLinker()
        else:
            self.term_linker = None

        self.cleaner = EntityCleaner(
            additional_cleaners=additional_cleaners,
        )

    def normalize(self, entity_sets: Sequence[DocEntity]) -> list[CanonicalEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
        """
        # removed_suppressed must be false to properly index against original terms
        terms: list[str] = [e.term for e in flatten(entity_sets)]
        normalized = self.cleaner.clean(terms, remove_suppressed=False)

        if self.term_linker is not None:
            links = self.term_linker.link(entity_sets)
        else:
            links = [None] * len(entity_sets)

        def get_canonical(
            entity: CanonicalEntity | None, normalized: str
        ) -> CanonicalEntity:
            # create a pseudo-canonical entity if no canonical entity found
            if entity is None or entity[1] is None:
                return CanonicalEntity(id="", name=normalized, aliases=[])
            return entity

        return [get_canonical(t[1], t[2]) for t in zip(terms, links, normalized)]

    def __call__(self, *args, **kwargs) -> list[CanonicalEntity]:
        return self.normalize(*args, **kwargs)
