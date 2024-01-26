from typing import Sequence
import logging
from prisma.enums import OntologyLevel
from prisma.models import Umls
from prisma.types import (
    UmlsCreateWithoutRelationsInput as UmlsCreateInput,
    UmlsUpdateInput,
)
from pydash import flatten

from constants.umls import DESIREABLE_ANCESTOR_TYPE_MAP
from data.domain.biomedical.umls import clean_umls_name
from utils.list import has_intersection

from .ancestor_selection import AncestorUmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS
from .types import UmlsInfo, compare_ontology_level


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsTransformer:
    def __init__(self, aui_lookup: dict[str, str]):
        # aui -> cui lookup (since hierarchy is stored as aui)
        self.aui_lookup: dict[str, str] = aui_lookup

    def __call__(self, r: dict) -> UmlsCreateInput:
        """
        Transform a single UMLS record for create
        """
        # reverse to get nearest ancestor first
        ancestors = r["hierarchy"].split(".")[::-1]
        ancestor_cuis = [self.aui_lookup.get(aui, "") for aui in ancestors]
        preferred_name = clean_umls_name(
            r["id"], r["name"], r["synonyms"], r["type_ids"], False
        )

        return UmlsCreateInput(
            **{
                **r,
                **{
                    f"l{i}_ancestor": ancestor_cuis[i] if i < len(ancestor_cuis) else ""
                    for i in range(MAX_DENORMALIZED_ANCESTORS)
                },  # type: ignore
            },
            rollup_id=r["id"],  # start with self as rollup
            preferred_name=preferred_name,
            level=OntologyLevel.UNKNOWN,
        )


class UmlsAncestorTransformer:
    """
    Class for transforming UMLS records
    """

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

        # generate all the level info first
        self.level_lookup = {
            r.id: UmlsInfo(
                id=r.id,
                name=r.name,
                level=self.umls_graph.get_ontology_level(r.id),
                type_ids=r.type_ids,
            )
            for r in records
        }

    @staticmethod
    def choose_best_ancestor(
        record: UmlsInfo,
        ancestors: tuple[UmlsInfo, ...],
    ) -> str:
        """
        Choose the best ancestor for the record
        - prefer ancestors that are *specific* genes/proteins/receptors, aka the drug's target, e.g. gpr83 (for instance)
        - avoid suppressed UMLS records
        - otherwise prefer just based on level (e.g. L1_CATEGORY, L2_CATEGORY, INSTANCE)

        Args:
            record (Umls): UMLS record
            ancestors (tuple[UmlsInfo]): ordered list of ancestors

        Returns (str): ancestor id, or "" if none found

        TODO:
        - prefer family rollups - e.g. serotonin for 5-HTXY receptors
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
            good_ancestors = [
                a
                for a in _ancestors
                if has_intersection(a.type_ids, good_ancestor_types)
            ]
            if len(good_ancestors) == 0:
                return None

            return choose_by_level(tuple(good_ancestors))

        def choose_by_level(_ancestors: tuple[UmlsInfo, ...]) -> str:
            """
            based on record level, find the best ancestor
            """
            # ancestors ABOVE the current level
            # e.g. L1_CATEGORY if record.level == INSTANCE
            # e.g. L2_CATEGORY if record.level == L1_CATEGORY
            ok_ancestors = [
                a
                for a in _ancestors
                if compare_ontology_level(a.level, record.level) > 0
            ]

            # if no ok ancestors, return ""
            if len(ok_ancestors) == 0:
                return ""

            # otherwise, use the first matching ancestor
            return ok_ancestors[0].id

        # prefer type-preferred ancestor, otherwise just go by level
        return choose_by_type(ancestors) or choose_by_level(ancestors)

    def transform(self, partial_record: Umls) -> UmlsUpdateInput:
        """
        Transform a single UMLS record with updates (level, rollup_id)
        """
        if self.level_lookup is None:
            raise ValueError("level_lookup is not initialized")

        ancestors = tuple(
            [
                self.level_lookup[partial_record.__dict__[f"l{i}_ancestor"]]
                for i in range(MAX_DENORMALIZED_ANCESTORS)
            ]
        )

        # add level info so it can be used for update and choose_best_ancestor
        record = UmlsInfo.from_umls(
            partial_record, level=self.level_lookup[partial_record.id].level
        )

        return UmlsUpdateInput(
            level=record.level,
            rollup_id=UmlsAncestorTransformer.choose_best_ancestor(record, ancestors),
        )
