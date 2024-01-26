from typing import Sequence
import logging
from networkx import DiGraph
from prisma.enums import OntologyLevel

from clients.low_level.prisma import prisma_client
from clients.umls.graph import UmlsGraph
from clients.umls.types import EdgeRecord, NodeRecord
from constants.umls import (
    MOST_PREFERRED_UMLS_TYPES,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
)
from data.etl.entity.biomedical_entity.umls.types import increment_ontology_level
from typings.documents.common import DocType
from utils.classes import overrides
from utils.re import get_or_re


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS = (
    "isa",
    "inverse_isa",
    "mapped_from",
    "mapped_to",
)
INSTANCE_THRESHOLD = 25
MAX_DEPTH = 2
MIN_PREV_COUNT = 5
OVERRIDE_DELTA = 500


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(self, doc_type: DocType = DocType.patent):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        # initialize superclass with _add_level_info transform
        super().__init__(transform_graph=AncestorUmlsGraph._add_level_info)
        self.doc_type = doc_type

    @classmethod
    async def create(cls, doc_type: DocType = DocType.patent):
        """
        Factory for AncestorUmlsGraph
        """
        aug = cls(doc_type)
        await aug.load()
        return aug

    @overrides(UmlsGraph)
    async def get_nodes(
        self,
        considered_tuis: list[str] = list(MOST_PREFERRED_UMLS_TYPES.keys()),
    ) -> list[NodeRecord]:
        """
        Query nodes from umls
        """
        query = rf"""
            SELECT umls.id as id, count(*) as count
            FROM biomedical_entity, intervenable, _entity_to_umls AS etu, umls
            WHERE biomedical_entity.id=intervenable.entity_id
            AND etu."A"=intervenable.id
            AND umls.id=etu."B"
            AND umls.type_ids && $1
            GROUP BY umls.id

            UNION

            -- indication-mapped UMLS terms associated with docs, if preferred type
            SELECT umls.id as id, count(*) as count
            FROM biomedical_entity, indicatable, _entity_to_umls AS etu, umls
            WHERE biomedical_entity.id=indicatable.entity_id
            AND etu."A"=indicatable.id
            AND umls.id=etu."B"
            AND umls.type_ids && $1
            GROUP BY umls.id
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, considered_tuis)
        return [NodeRecord(**r) for r in results]

    @overrides(UmlsGraph)
    async def get_edges(
        self,
        considered_relationships: tuple[str, ...] = DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS,
        considered_tuis: tuple[str, ...] = tuple(MOST_PREFERRED_UMLS_TYPES.keys()),
        name_suppressions: Sequence[str] = UMLS_NAME_SUPPRESSIONS,
        cui_suppressions: Sequence[str] = tuple(UMLS_CUI_SUPPRESSIONS.keys()),
    ) -> list[EdgeRecord]:
        """
        Query edges from umls
        - non-null hierarchy (good idea??)
        - only isa/inverse_isa/mapped_from/mapped_to or no relationship
        - limits types to biomedical
        - applies some naming restrictions (via 'suppressions')
        - suppresses entities whose name is also a type (indicates overly general)

        Query took 4.51 minutes (depth < 3)
        """
        # head == parent, tail == child
        query = rf"""
            -- patent to umls edges
            WITH RECURSIVE working_terms AS (
                SELECT
                    distinct umls.id as head,
                    'leaf' as tail,
                    1 AS depth
                FROM biomedical_entity, intervenable, _entity_to_umls AS etu, umls
                WHERE biomedical_entity.id=intervenable.entity_id
                AND etu."A"=intervenable.id
                AND umls.id=etu."B"
                AND umls.type_ids && $1

                UNION

                -- indication-mapped UMLS terms associated with docs, if preferred type
                SELECT
                    distinct umls.id as head,
                    'leaf' as tail,
                    1 AS depth
                FROM biomedical_entity, indicatable, _entity_to_umls AS etu, umls
                WHERE biomedical_entity.id=indicatable.entity_id
                AND etu."A"=indicatable.id
                AND umls.id=etu."B"
                AND umls.type_ids && $1

                UNION

                -- UMLS to UMLS relationships
                -- e.g. C12345 & C67890
                SELECT
                    head_id as head,
                    tail_id as tail,
                    ut.depth + 1
                FROM umls_graph
                JOIN working_terms ut ON ut.tail = head_id
                JOIN umls as head_entity on head_entity.id = umls_graph.head_id
                JOIN umls as tail_entity on tail_entity.id = umls_graph.tail_id
                where ut.depth <= {MAX_DEPTH}
                and (
                    relationship is null
                    OR relationship in {considered_relationships}
                )
                and head_entity.type_ids && $1
                and tail_entity.type_ids && $1
                and head_id not in {cui_suppressions}
                and tail_id not in {cui_suppressions}
                and ts_lexize('english_stem', head_entity.type_names[1]) <> ts_lexize('english_stem', head_name)  -- exclude entities with a name that is also the type
                and ts_lexize('english_stem', tail_entity.type_names[1]) <> ts_lexize('english_stem', tail_name)
                and not head_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
                and not tail_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
            )
            SELECT DISTINCT head, tail
            FROM working_terms;
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, considered_tuis)
        return [EdgeRecord(**r) for r in results]

    @staticmethod
    def _propagate_counts(G: DiGraph) -> DiGraph:
        """
        Propagate counts up the tree
            - assumes leafs have counts
            - assumes edges are directed from child to parent
            - has parent counts = sum of children counts

        See distribution of counts:
        ```
        import seaborn as sns
        from data.etl.entity.biomedical_entity.umls.ancestor_selection import AncestorUmlsGraph
        g = await AncestorUmlsGraph.create()
        data = [v["count"] for v in g.nodes.values()]
        sns.displot(data, kde=True, aspect=10/4)
        ```
        """
        logger.info("Recursively propagating counts up the tree")

        def _propagate(g: DiGraph, node_id: str):
            child_ids: list[str] = list(g.successors(node_id))

            # for children with no counts, aka non-leaf nodes, recurse
            # e.g. if we're on Grandparent1, Parent1 and Parent2 have no counts,
            # so we recurse to set Parent1 and Parent2 counts based on their children
            for child_id in child_ids:
                if g.nodes[child_id].get("count") is None:
                    _propagate(g, child_id)

            # set count to sum of all children counts
            g.nodes[node_id]["count"] = sum(
                g.nodes[child_id]["count"] for child_id in child_ids
            )

        # take all nodes with no *incoming* edges, i.e. root nodes
        for node_id in [n for n, d in G.in_degree() if d == 0]:
            _propagate(G, node_id)

        return G

    @staticmethod
    def _add_level_info(G: DiGraph) -> DiGraph:
        """
        Add ontology level to nodes

        Calls _propagate_counts to set counts on all nodes first.
        """

        def get_level(
            current_node: NodeRecord,
            prev_count: int | None = None,
            last_level: OntologyLevel | None = None,
            max_parent_count: int | None = None,
        ) -> OntologyLevel:
            current_count = current_node.count or 0

            if prev_count is None:
                """
                leaf node. level it INSTANCE if sufficiently common.
                """
                if current_count < INSTANCE_THRESHOLD:
                    return OntologyLevel.SUBINSTANCE

                return OntologyLevel.INSTANCE

            if last_level is None:
                raise ValueError(
                    "last_level must be provided if prev_count is provided"
                )

            if max_parent_count is None:
                """
                root node. increment level
                """
                return increment_ontology_level(last_level)

            # Else, we're in the middle of the tree and will increment the ancestor level if:
            # 1) the absolute change in counts between prev and current is greater than OVERRIDE_DELTA, or
            # 2) the rate of change is *increasing*
            #       reasoning: if rate of change ↓↓↓, grandparent isn't so different than the parent
            #       tl;dr we're trying to get rid of the useless middle managers of the UMLS ontology
            parent_current_delta = max_parent_count - current_count
            current_prev_delta = current_count - prev_count

            if current_prev_delta > OVERRIDE_DELTA or (
                parent_current_delta > current_prev_delta
                and prev_count > MIN_PREV_COUNT  # avoid big changes in small numbers
            ):
                return increment_ontology_level(last_level)

            return OntologyLevel.NA

        def set_level(
            _G: DiGraph,
            node: NodeRecord,
            prev_node: NodeRecord | None = None,
            last_level: OntologyLevel | None = None,
        ) -> None:
            """
            Set level on node, and recurse through parents
            (mutation!)
            """
            parent_ids = list(_G.predecessors(node.id))
            max_parent_count = (
                max([_G.nodes[p]["count"] for p in parent_ids]) if parent_ids else None
            )
            prev_count = prev_node.count if prev_node else None
            level = get_level(node, prev_count, last_level, max_parent_count)
            _G.nodes[node.id]["level"] = level

            # last real level as basis for inc
            new_last_level = level if level != OntologyLevel.NA else last_level

            # recurse through parents
            for parent_id in parent_ids:
                parent = NodeRecord(**_G.nodes[parent_id])
                set_level(_G, parent, NodeRecord(**_G.nodes[node.id]), new_last_level)

        logger.info("Recursively propagating counts up the tree")

        # propogate counts up the tree
        new_g = AncestorUmlsGraph._propagate_counts(G.copy().to_directed())

        # take all nodes with no *outgoing* edges, i.e. leaf nodes
        for node_id in [n for n, d in new_g.out_degree() if d == 0]:
            node = new_g.nodes[node_id]
            set_level(new_g, NodeRecord(**node))

        return new_g

    def get_ontology_level(self, cui: str) -> OntologyLevel:
        """
        Get ontology level for cui
        """
        if not cui in self.nodes:
            logger.warning("%s not in graph", cui)
            return OntologyLevel.UNKNOWN

        node = NodeRecord(**self.nodes[cui])

        return node.level

    def get_count(self, cui: str) -> int | None:
        """
        Get count for cui
        """
        if not cui in self.nodes:
            logger.warning("%s not in graph", cui)
            return None

        node = NodeRecord(**self.nodes[cui])

        return node.count
