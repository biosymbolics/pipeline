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

    def normalize(self, entity_sets: Sequence[DocEntity]) -> list[DocEntity]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (Sequence[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
        """
        # removed_suppressed must be false to properly index against original terms
        cleaned_entity_sets = self.cleaner.clean(entity_sets, remove_suppressed=False)
        print("CLEANED", cleaned_entity_sets)

        if self.term_linker is not None:
            return self.term_linker.link(cleaned_entity_sets)

        return cleaned_entity_sets

    def __call__(self, *args, **kwargs) -> list[DocEntity]:
        return self.normalize(*args, **kwargs)
