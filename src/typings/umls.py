from dataclasses import dataclass
from typing import Literal

from utils.classes import ByDefinitionOrderEnum


@dataclass(frozen=True)
class UmlsRecord:
    def __getitem__(self, item):
        return getattr(self, item)

    id: str
    canonical_name: str
    num_descendants: int
    hierarchy: str | None
    type_id: str
    type_name: str


class OntologyLevel(ByDefinitionOrderEnum):
    GENERAL_CATEGORY = "GENERAL_CATEGORY"  # least specific
    CATEGORY = "CATEGORY"
    INSTANCE = "INSTANCE"  # most specific
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find(
        cls,
        id: str,
        betweenness_map: dict[str, float],
    ):
        """
        Simple heuristic to find approximate semantic level of UMLS record
        TODO: make this a model
        """
        if id not in betweenness_map:
            # assume it isn't in the map due to too low degree
            return cls.INSTANCE

        if betweenness_map[id] > 0.005:
            # 163 as of 11/23
            return cls.GENERAL_CATEGORY

        if betweenness_map[id] > 0.0001:
            # 6418 as of 11/23
            return cls.CATEGORY

        # 49837 as of 11/23
        return cls.INSTANCE


@dataclass(frozen=True)
class UmlsLookupRecord(UmlsRecord):
    preferred_name: str
    level: OntologyLevel
    instance_rollup: str | None
    category_rollup: str | None


RollupLevel = Literal[OntologyLevel.INSTANCE, OntologyLevel.INSTANCE]  # type: ignore
