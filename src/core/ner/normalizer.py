from core.ner.cleaning import EntityCleaner
from core.ner.linker import TermLinker
from core.ner.types import CanonicalEntity


class TermNormalizer:
    """
    Normalizes and attempts to link entities.
    If no canonical entity found, then normalized term is returned
    Note that original term should always be preserved in order to keep association to original source.
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
