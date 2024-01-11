from typing import Callable, Literal

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


RollupLevel = Literal[OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY]  # type: ignore
