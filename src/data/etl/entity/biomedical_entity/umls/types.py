from pydantic import BaseModel
from prisma.enums import OntologyLevel
from prisma.models import Umls


ONTOLOGY_LEVEL_MAP = {
    OntologyLevel.SUBINSTANCE: 0,
    OntologyLevel.INSTANCE: 1,
    OntologyLevel.L1_CATEGORY: 2,
    OntologyLevel.L2_CATEGORY: 3,
    # OntologyLevel.NA: -1, # excluded
    # OntologyLevel.UNKNOWN: -1, # excluded
}


def compare_ontology_level(a: OntologyLevel, b: OntologyLevel) -> int:
    """
    Compare two ontology levels

    Higher means more general.

    Returns:
        positive if a > b
        0 if a == b
        negative if a < b
    """
    if a not in ONTOLOGY_LEVEL_MAP and b not in ONTOLOGY_LEVEL_MAP:
        return 0
    if a not in ONTOLOGY_LEVEL_MAP:
        return -1
    if b not in ONTOLOGY_LEVEL_MAP:
        return 1
    return ONTOLOGY_LEVEL_MAP[a] - ONTOLOGY_LEVEL_MAP[b]


def increment_ontology_level(level: OntologyLevel) -> OntologyLevel:
    if level == OntologyLevel.SUBINSTANCE:
        return OntologyLevel.INSTANCE
    if level == OntologyLevel.INSTANCE:
        return OntologyLevel.L1_CATEGORY
    if level == OntologyLevel.L1_CATEGORY:
        return OntologyLevel.L2_CATEGORY

    raise ValueError(f"Cannot increment level {level}")


class UmlsInfo(BaseModel, object):
    id: str
    name: str
    level: OntologyLevel
    type_ids: list[str] = []

    @staticmethod
    def from_umls(umls: Umls, **kwargs) -> "UmlsInfo":
        # combine with kwargs
        _umls = Umls(**{**umls.__dict__, **kwargs})
        return UmlsInfo(
            id=_umls.id,
            name=_umls.name,
            level=umls.level,
            type_ids=_umls.type_ids,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UmlsInfo):
            return NotImplemented
        return compare_ontology_level(self.level, other.level) == 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, UmlsInfo):
            return NotImplemented
        return compare_ontology_level(self.level, other.level) < 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, UmlsInfo):
            return NotImplemented
        return compare_ontology_level(self.level, other.level) > 0

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, UmlsInfo):
            return NotImplemented
        return compare_ontology_level(self.level, other.level) >= 0

    def __le__(self, other: object) -> bool:
        if not isinstance(other, UmlsInfo):
            return NotImplemented
        return compare_ontology_level(self.level, other.level) <= 0
