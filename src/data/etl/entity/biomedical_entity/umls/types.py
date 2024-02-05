from pydantic import BaseModel
from prisma.enums import OntologyLevel
from prisma.models import Umls

from .constants import ONTOLOGY_LEVEL_MAP


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


class UmlsInfo(BaseModel, object):
    id: str
    name: str
    count: int
    level: OntologyLevel
    type_ids: list[str] = []
