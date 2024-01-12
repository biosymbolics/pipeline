from typing import Callable, Literal
from prisma.enums import OntologyLevel

L1_CATEGORY_CUTOFF = 0.0001


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
