from dataclasses import asdict, dataclass
from typing import Literal

from utils.classes import ByDefinitionOrderEnum


class OntologyLevel(ByDefinitionOrderEnum):
    L2_CATEGORY = "L2_CATEGORY"  # least specific
    L1_CATEGORY = "L1_CATEGORY"
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
        """
        if id not in betweenness_map:
            # assume it isn't in the map due to too low degree
            return cls.INSTANCE

        if betweenness_map[id] < 0.0001:
            # 49837 as of 11/23
            return cls.L1_CATEGORY

        # 6418 as of 11/23
        return cls.L2_CATEGORY


@dataclass(frozen=True)
class UmlsRecord:
    def __getitem__(self, item):
        return getattr(self, item)

    def keys(self):
        return self.__dataclass_fields__.keys()

    def values(self):
        return self.__dataclass_fields__.values()

    def items(self):
        return self.__dataclass_fields__.items()

    id: str
    canonical_name: str
    num_descendants: int
    hierarchy: str | None
    type_id: str
    type_name: str


@dataclass(frozen=True)
class IntermediateUmlsRecord(UmlsRecord):
    preferred_name: str
    level: OntologyLevel
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
    synonyms: list[str]


@dataclass(frozen=True)
class UmlsLookupRecord(IntermediateUmlsRecord):
    instance_rollup: str | None
    category_rollup: str | None

    @classmethod
    def from_intermediate(cls, ir: IntermediateUmlsRecord, **kwargs):
        is_data_class = getattr(ir, "__dataclass_fields__", None) is not None

        if is_data_class:
            return cls(**asdict(ir), **kwargs)

        return cls(**{**ir, **kwargs})  # type: ignore


RollupLevel = Literal[OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY]  # type: ignore
