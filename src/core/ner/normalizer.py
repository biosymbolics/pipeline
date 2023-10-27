"""
Linking/cleaning of terms
"""
from typing import Sequence
from core.ner.cleaning import CleanFunction, EntityCleaner
from core.ner.linker import TermLinker
from core.ner.types import CanonicalEntity


class TermNormalizer:
    """
    Normalizes and attempts to link entities.
    If no canonical entity found, then normalized term is returned
    Note that original term should always be preserved in order to keep association to original source.

    Usage:
        normalizer = TermNormalizer()
        terms = normalizer.normalize([
            "Tipranavir (TPV)",
            "BILR 355 - D4",
            "bivatuzumab mertansine",
            "BIBT 986 BS - single rising dose",
            "RDEA3170 10 mg",
            "Minoxidil Solution 5%",
            "ASP2151 400mg + 100mg ciclosporin",
            "Misoprostol 600 mcg 90 minutes prior to procedure",
            "BI 409306 10 mg QD",
            "DA-5204",
            "SAGE-547",
            "GSK2838232 PIB (API)",
            "Revusiran (ALN-TTRSC)",
            "Placebo matching atosiban",
        ])
        [(t[0], t[1].name) for t in terms]
    """

    def __init__(
        self,
        link: bool = True,
        additional_cleaners: Sequence[CleanFunction] = [],
        additional_removal_terms: Sequence[str] = [],
    ):
        """
        Initialize term normalizer using existing model
        """
        if link:
            self.term_linker: TermLinker | None = TermLinker()
        else:
            self.term_linker = None
        self.cleaner: EntityCleaner = EntityCleaner(
            additional_removal_terms=additional_removal_terms,
            additional_cleaners=additional_cleaners,
        )

    def normalize(self, terms: list[str]) -> list[tuple[str, CanonicalEntity]]:
        """
        Normalize and link terms to canonical entities

        Args:
            terms (list[str]): list of terms to normalize

        Note:
            - canonical linking is based on normalized term
            - if no linking is found, then normalized term is as canonical_name, with an empty id
            - will return fewer terms than input, if term meets conditions for suppression
        """
        # removed_suppressed must be false to properly index against original terms
        normalized = self.cleaner.clean(terms, remove_suppressed=False)

        if self.term_linker is not None:
            links = self.term_linker.link(normalized)
        else:
            links = [None] * len(normalized)

        def get_canonical(
            entity: tuple[str, CanonicalEntity | None] | None, normalized: str
        ) -> CanonicalEntity:
            if entity is None or entity[1] is None:
                # create a pseudo-canonical entity
                return CanonicalEntity(id="", name=normalized, aliases=[])
            return entity[1]

        tups = [
            (t[0], get_canonical(t[1], t[2])) for t in zip(terms, links, normalized)
        ]
        return tups

    def __call__(self, *args, **kwargs) -> list[tuple[str, CanonicalEntity]]:
        return self.normalize(*args, **kwargs)
