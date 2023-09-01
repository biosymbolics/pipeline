from core.ner.cleaning import EntityCleaner
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

    def __init__(self):
        """
        Initialize term normalizer using existing model
        """
        self.term_linker: TermLinker = TermLinker()
        self.cleaner: EntityCleaner = EntityCleaner()

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
        normalized = self.cleaner.clean(terms)
        links = self.term_linker.link(normalized)

        def get_canonical(
            entity: tuple[str, CanonicalEntity | None], normalized: str
        ) -> CanonicalEntity:
            if entity[1] is None:
                # create a pseudo-canonical entity
                return CanonicalEntity(id="", name=normalized, aliases=[])
            return entity[1]

        tups = [
            (t[0], get_canonical(t[1], t[2]))
            for t in zip(terms, links, normalized)
            if len(t[0]) > 0
        ]
        return tups

    def __call__(self, *args, **kwargs) -> list[tuple[str, CanonicalEntity]]:
        return self.normalize(*args, **kwargs)
