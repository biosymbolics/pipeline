import time
from typing import Sequence
import logging
from pydantic import BaseModel
from prisma.models import Umls
from prisma.types import (
    UmlsCreateWithoutRelationsInput as UmlsCreateInput,
    UmlsUpdateInput,
)

from data.domain.biomedical.umls import clean_umls_name
from typings.umls import OntologyLevel, get_ontology_level

from .ancestor_selection import AncestorUmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsIdLevel(BaseModel):
    id: str
    level: OntologyLevel


class UmlsTransformer:
    def __init__(self, aui_lookup: dict[str, str]):
        # aui -> cui lookup (since hierarchy is stored as aui)
        self.aui_lookup: dict[str, str] = aui_lookup

    def __call__(self, r: dict) -> UmlsCreateInput:
        """
        Transform a single UMLS record for create
        """
        # reverse to get nearest ancestor first
        ancestors = (r.get("hierarchy") or "").split(".")[::-1]
        ancestor_cuis = [self.aui_lookup.get(aui, "") for aui in ancestors]
        preferred_name = clean_umls_name(
            r["id"],
            r["name"],
            r.get("synonyms") or [],
            r.get("type_ids") or [],
            False,
        )
        return UmlsCreateInput(
            **{
                **r,
                **{
                    f"l{i}_ancestor": ancestor_cuis[i] if i < len(ancestor_cuis) else ""
                    for i in range(MAX_DENORMALIZED_ANCESTORS)
                },  # type: ignore
                "level": OntologyLevel.UNKNOWN,
                "preferred_name": preferred_name,
            }
        )


class UmlsLevelTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(self):
        self.lookup_dict: dict[str, UmlsIdLevel] | None = None

    @staticmethod
    async def create() -> "UmlsLevelTransformer":
        """
        Factory for UmlsLevelTransformer
        """
        ult = UmlsLevelTransformer()
        await ult.load()
        return ult

    async def load(self):
        """
        Load UMLS graph
        """
        self.umls_graph = await AncestorUmlsGraph.create()

    @staticmethod
    def find_level_ancestor(
        record: UmlsIdLevel,
        levels: Sequence[OntologyLevel],
        ancestors: tuple[UmlsIdLevel, ...],
    ) -> str:
        """
        Find first ancestor at the specified level

        Args:
            record (Umls): UMLS record
            level (OntologyLevel): level to find
            ancestors (tuple[UmlsIdLevel]): ordered list of ancestors

        Returns (str): ancestor id, or "" if none found
        """
        level_ancestors = [a for a in ancestors if a.level in levels]

        if len(level_ancestors) == 0:
            # return self as instance ancestor if no ancestors
            if record.level in levels:
                return record.id

            return ""

        # for instance level, use the last "INSTANCE" ancestor
        if OntologyLevel.INSTANCE in levels:
            return level_ancestors[-1].id

        # else, use self as ancestor if matching or exceeding level
        if all(record.level >= level for level in levels):
            return record.id

        # otherwise, use the first matching ancestor
        return level_ancestors[0].id

    def transform(self, r: Umls) -> UmlsUpdateInput:
        """
        Transform a single UMLS record with updates (level, instance_rollup, category_rollup)
        """
        if self.lookup_dict is None:
            raise ValueError("Lookup dict is not initialized")

        cuis = [r.__dict__[f"l{i}_ancestor"] for i in range(MAX_DENORMALIZED_ANCESTORS)]
        ancestors = tuple(
            [self.lookup_dict[cui] for cui in cuis if cui in self.lookup_dict]
        )
        level = get_ontology_level(r.id, self.umls_graph.get_umls_centrality)
        id_level = UmlsIdLevel(id=r.id, level=level)
        return UmlsUpdateInput(
            level=level,
            instance_rollup=UmlsLevelTransformer.find_level_ancestor(
                id_level, [OntologyLevel.INSTANCE], ancestors
            ),
            category_rollup=UmlsLevelTransformer.find_level_ancestor(
                id_level,
                [OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY],
                ancestors,
            ),
        )
