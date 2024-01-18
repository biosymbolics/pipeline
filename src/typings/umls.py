from typing import Callable, Literal
from prisma.enums import OntologyLevel

L1_CATEGORY_CUTOFF = 0.0001

ONTOLOGY_LEVEL_MAP = {
    OntologyLevel.INSTANCE: 1,
    OntologyLevel.L1_CATEGORY: 2,
    OntologyLevel.L2_CATEGORY: 3,
    # OntologyLevel.NA: -1, # excluded
    # OntologyLevel.UNKNOWN: -1, # excluded
}


def compare_ontology_levels(a: OntologyLevel, b: OntologyLevel) -> int:
    """
    Compare two ontology levels

    Returns:
        positive if a > b
        0 if a == b
        negative if a < b
    """
    if a not in ONTOLOGY_LEVEL_MAP and b not in ONTOLOGY_LEVEL_MAP:
        return 0
    if a not in ONTOLOGY_LEVEL_MAP:
        return 1
    if b not in ONTOLOGY_LEVEL_MAP:
        return -1
    return ONTOLOGY_LEVEL_MAP[a] - ONTOLOGY_LEVEL_MAP[b]


def get_ontology_level(
    id: str,
    get_centrality: Callable[[str], float],
):
    """
    Simple heuristic to find approximate semantic level of UMLS record
    """
    centrality = get_centrality(id)

    if centrality == -1:
        return OntologyLevel.NA  # not eligible for inclusion

    if centrality == 0:
        # assume it isn't in the map due to too low degree
        return OntologyLevel.INSTANCE

    if centrality < L1_CATEGORY_CUTOFF:
        # 49837 as of 11/23
        return OntologyLevel.L1_CATEGORY

    # 6418 as of 11/23
    return OntologyLevel.L2_CATEGORY


RollupLevel = Literal[OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY]  # type: ignore
