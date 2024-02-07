from typing import Sequence
import logging
from prisma.models import Umls
from prisma.types import UmlsUpdateInput
from pydash import compact, flatten, omit


from .ancestor_selection import AncestorUmlsGraph
from .types import UmlsInfo, compare_ontology_level
from .utils import choose_best_available_ancestor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsAncestorTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(self, umls_graph: AncestorUmlsGraph):
        self.umls_graph = umls_graph

    @staticmethod
    async def create() -> "UmlsAncestorTransformer":
        """
        Factory for UmlsAncestorTransformer
        """
        g = await AncestorUmlsGraph.create()
        return UmlsAncestorTransformer(g)

    @staticmethod
    def choose_best_ancestor(
        child: UmlsInfo,
        ancestors: Sequence[UmlsInfo],
    ) -> str:
        """
        Choose the best ancestor for the child
        - prefer ancestors that are *specific* genes/proteins/receptors, aka the drug's target, e.g. gpr83 (for instance)
        - avoid suppressed UMLS records
        - require level be higher than child

        Args:
            child (UmlsInfo): UMLS info
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
            a for a in ancestors if compare_ontology_level(a.level, child.level) > 0
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

    def transform(self, record: Umls) -> UmlsUpdateInput | None:
        """
        Transform a single UMLS record with updates (level, rollup_id)
        """

        G = self.umls_graph.G

        parent_ids = list(G.predecessors(record.id))
        # grandparent_ids = flatten(
        #     [list(G.predecessors(parent_id)) for parent_id in parent_ids]
        # )
        ancestor_ids = parent_ids  # + grandparent_ids

        ancestors = compact(
            [UmlsInfo(**G.nodes[ancestor_id]) for ancestor_id in ancestor_ids]
        )

        if not G.nodes.get(record.id):
            return None  # irrelevant record

        # get record with updated level info
        updated_record = UmlsInfo(**omit(G.nodes[record.id], "level_override"))

        return UmlsUpdateInput(
            id=record.id,
            count=updated_record.count,
            level=updated_record.level,
            rollup_id=UmlsAncestorTransformer.choose_best_ancestor(
                updated_record, ancestors
            ),
        )
