from typing import Sequence
import logging
from prisma.models import Umls
from prisma.types import UmlsUpdateInput
from pydash import compact


from .ancestor_selection import AncestorUmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS
from .types import UmlsInfo, compare_ontology_level
from .utils import choose_best_available_ancestor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsAncestorTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(
        self, umls_graph: AncestorUmlsGraph, level_lookup: dict[str, UmlsInfo]
    ):
        self.umls_graph = umls_graph
        self.level_lookup: dict[str, UmlsInfo] = level_lookup

    @staticmethod
    async def create(records: Sequence[Umls]) -> "UmlsAncestorTransformer":
        """
        Factory for UmlsAncestorTransformer
        """
        g, level_lookup = await UmlsAncestorTransformer.load(records)
        return UmlsAncestorTransformer(g, level_lookup)

    @staticmethod
    async def load(
        records: Sequence[Umls],
    ) -> tuple[AncestorUmlsGraph, dict[str, UmlsInfo]]:
        """
        Load UMLS graph & level lookup info
        """
        g = await AncestorUmlsGraph.create()

        # only includes known nodes
        level_lookup = {
            r.id: UmlsInfo(
                id=r.id,
                name=r.name,
                count=g.get_count(r.id) or 0,
                level=g.get_ontology_level(r.id),
                type_ids=r.type_ids,
            )
            for r in records
            if g.has_node(r.id)
        }
        return g, level_lookup

    @staticmethod
    def choose_best_ancestor(
        child: UmlsInfo,
        ancestors: tuple[UmlsInfo, ...],
    ) -> str:
        """
        Choose the best ancestor for the child
        - prefer ancestors that are *specific* genes/proteins/receptors, aka the drug's target, e.g. gpr83 (for instance)
        - avoid suppressed UMLS records
        - require level be higher than child

        Args:
            record (UmlsInfo): UMLS info
            ancestors (tuple[UmlsInfo]): ordered list of ancestors

        Returns (str): ancestor id

        TODO:
        - prefer family rollups - e.g. serotonin for 5-HTXY receptors
        - know that "inhibitor", "antagonist", "agonist" etc are children of "modulator"

        NOTES:
        - https://www.d.umn.edu/~tpederse/Tutorials/IHI2012-semantic-similarity-tutorial.pdf has good concepts to apply
            such as "least common subsumer" and information-based mutual information measures
        """

        # ancestors eligible if the level is higher than child
        eligible_ancestors = [
            a
            for i, a in enumerate(ancestors)
            if compare_ontology_level(a.level, child.level) > 0
            # only consider monotonic levels (avoiding a confused random SUBINSTANCE from being chosen if type == NA)
            and (i == 0 or compare_ontology_level(a.level, ancestors[i - 1].level) >= 0)
        ]

        logger.debug("Eligible ancestors for %s: %s", child.id, eligible_ancestors)

        # if no eligible ancestors, return self
        if len(eligible_ancestors) == 0:
            return child.id

        # if no type_ids, return first eligible ancestor
        if len(child.type_ids) == 0:
            logger.info(
                "No type Ids for %s; returning %s", child.id, eligible_ancestors[0].id
            )
            return eligible_ancestors[0].id

        best_ancestor = choose_best_available_ancestor(
            child.type_ids, eligible_ancestors
        )

        logger.debug("Best available ancestor for %s: %s", child.id, best_ancestor)

        if best_ancestor is None:
            return child.id

        return best_ancestor.id

    def transform(self, partial_record: Umls) -> UmlsUpdateInput | None:
        """
        Transform a single UMLS record with updates (level, rollup_id)
        """
        if self.level_lookup is None:
            raise ValueError("level_lookup is not initialized")

        ancestors = tuple(
            compact(
                [
                    self.level_lookup.get(partial_record.__dict__[f"l{i}_ancestor"])
                    for i in range(MAX_DENORMALIZED_ANCESTORS)
                ]
            )
        )

        # get record with updated level info
        record = self.level_lookup.get(partial_record.id)

        if not record:
            return None  # irrelevant record

        return UmlsUpdateInput(
            count=record.count,
            level=record.level,
            rollup_id=UmlsAncestorTransformer.choose_best_ancestor(record, ancestors),
        )
