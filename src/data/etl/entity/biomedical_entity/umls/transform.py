from typing import Sequence
import logging
from pydantic import BaseModel
from prisma.models import Umls
from prisma.types import (
    UmlsCreateWithoutRelationsInput as UmlsCreateInput,
    UmlsUpdateInput,
)
from pydash import flatten

from constants.umls import DESIREABLE_ANCESTOR_TYPE_MAP
from data.domain.biomedical.umls import clean_umls_name
from typings.umls import OntologyLevel, get_ontology_level
from utils.list import has_intersection

from .ancestor_selection import AncestorUmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsInfo(BaseModel):
    id: str
    level: OntologyLevel
    type_ids: Sequence[str] = []


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


class UmlsAncestorTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(self):
        self.level_lookup: dict[str, UmlsInfo] | None = None

    @staticmethod
    async def create(records: Sequence[Umls]) -> "UmlsAncestorTransformer":
        """
        Factory for UmlsAncestorTransformer
        """
        ult = UmlsAncestorTransformer()
        await ult.load(records)
        return ult

    async def load(self, records: Sequence[Umls]):
        """
        Load UMLS graph
        """
        self.umls_graph = await AncestorUmlsGraph.create()
        self.ancestor_lookup = {
            r.id: UmlsInfo(
                id=r.id,
                level=get_ontology_level(r.id, self.umls_graph.get_umls_centrality),
                type_ids=r.type_ids,
            )
            for r in records
        }

    @staticmethod
    def choose_best_ancestor(
        record: UmlsInfo,
        levels: Sequence[OntologyLevel],  # acceptable/desired level(s)
        ancestors: tuple[UmlsInfo, ...],
    ) -> str:
        """
        Choose the best ancestor for the record at the specified level(s)
        - prefer ancestors that are *specific* genes/proteins/receptors, aka the drug's target, e.g. gpr83 (for instance)
        - otherwise prefer just based on level (e.g. L1_CATEGORY, L2_CATEGORY, INSTANCE)

        Args:
            record (Umls): UMLS record
            levels (OntologyLevel): acceptable/desired level(s)
            ancestors (tuple[UmlsInfo]): ordered list of ancestors

        Returns (str): ancestor id, or "" if none found

        TODO:
        - prefer family rollups - e.g. serotonin for 5-HTXY receptors
        - prefer ancestors with the most ontology representation (e.g. MESH and NCI and SNOMED over just MESH)
        - know that "inhibitor", "antagonist", "agonist" etc are children of "modulator"

        NOTES:
        - https://www.d.umn.edu/~tpederse/Tutorials/IHI2012-semantic-similarity-tutorial.pdf has good concepts to apply
            such as "least common subsumer" and information-based mutual information measures
        """

        def choose_by_type(_ancestors: tuple[UmlsInfo, ...]) -> str | None:
            """
            based on record types, find desireable ancestor type(s)
            """
            good_ancestor_types = flatten(
                [
                    DESIREABLE_ANCESTOR_TYPE_MAP.get(type_id, [])
                    for type_id in record.type_ids
                ]
            )
            ok_ancestors = [
                a
                for a in _ancestors
                if has_intersection(a.type_ids, good_ancestor_types)
            ]
            if len(ok_ancestors) == 0:
                return None

            return choose_by_level(tuple(ok_ancestors))

        def choose_by_level(_ancestors: tuple[UmlsInfo, ...]) -> str:
            """
            based on record level, find the first ancestor at or above the desired level
            """
            ok_ancestors = [a for a in _ancestors if a.level in levels]

            # return self as instance ancestor if no ancestors
            # AND record level is greater than or equal to any desired level
            if len(ok_ancestors) == 0:
                if any(record.level >= level for level in levels):
                    return record.id

                return ""

            # for instance level, use the last "INSTANCE" ancestor
            if OntologyLevel.INSTANCE in levels:
                return ok_ancestors[-1].id

            # else, use self as ancestor if matching or exceeding level
            if all(record.level >= level for level in levels):
                return record.id

            # otherwise, use the first matching ancestor
            return ok_ancestors[0].id

        # prefer type-preferred ancestor, otherwise just go by level
        return choose_by_type(ancestors) or choose_by_level(ancestors)

    def transform(self, r: Umls) -> UmlsUpdateInput:
        """
        Transform a single UMLS record with updates (level, instance_rollup, category_rollup)
        """
        if self.ancestor_lookup is None:
            raise ValueError("ancestor_lookup is not initialized")

        cuis = [r.__dict__[f"l{i}_ancestor"] for i in range(MAX_DENORMALIZED_ANCESTORS)]
        ancestors = tuple(
            [self.ancestor_lookup[cui] for cui in cuis if cui in self.ancestor_lookup]
        )
        level = get_ontology_level(r.id, self.umls_graph.get_umls_centrality)
        id_level = UmlsInfo(id=r.id, level=level, type_ids=r.type_ids)
        return UmlsUpdateInput(
            level=level,
            instance_rollup_id=UmlsAncestorTransformer.choose_best_ancestor(
                id_level, [OntologyLevel.INSTANCE], ancestors
            ),
            category_rollup_id=UmlsAncestorTransformer.choose_best_ancestor(
                id_level,
                [OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY],
                ancestors,
            ),
        )
