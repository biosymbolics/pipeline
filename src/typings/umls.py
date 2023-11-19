from dataclasses import dataclass
from typing import Callable, Literal

from typings.core import Dataclass

from utils.classes import ByDefinitionOrderEnum

L1_CATEGORY_CUTOFF = 0.0001


class OntologyLevel(ByDefinitionOrderEnum):
    INSTANCE = "INSTANCE"  # most specific
    L1_CATEGORY = "L1_CATEGORY"
    L2_CATEGORY = "L2_CATEGORY"  # least specific
    NA = "NA"  # not a valid level

    @classmethod
    def find(
        cls,
        id: str,
        get_centrality: Callable[[str], float],
    ):
        """
        Simple heuristic to find approximate semantic level of UMLS record
        """
        centrality = get_centrality(id)

        if centrality == -1:
            return cls.NA  # not eligible for inclusion

        if centrality == 0:
            # assume it isn't in the map due to too low degree
            return cls.INSTANCE

        if centrality < L1_CATEGORY_CUTOFF:
            # 49837 as of 11/23
            return cls.L1_CATEGORY

        # 6418 as of 11/23
        return cls.L2_CATEGORY


@dataclass(frozen=True)
class UmlsRecord(Dataclass):
    id: str
    canonical_name: str
    category_rollup: str | None
    hierarchy: str | None
    instance_rollup: str | None
    num_descendants: int
    level: OntologyLevel
    preferred_name: str
    synonyms: list[str]
    type_id: str
    type_name: str
    l0_ancestor: str | None
    l1_ancestor: str | None
    l2_ancestor: str | None
    l3_ancestor: str | None
    l4_ancestor: str | None
    l5_ancestor: str | None
    l6_ancestor: str | None
    l7_ancestor: str | None
    l8_ancestor: str | None
    l9_ancestor: str | None


RollupLevel = Literal[OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY]  # type: ignore
