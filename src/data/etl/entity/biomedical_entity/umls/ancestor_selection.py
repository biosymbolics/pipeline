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

"""
Debugging
select be.canonical_id, be.name, be.count, parent_be.name from biomedical_entity be, _entity_to_parent etp, biomedical_entity parent_be where etp."B"=be.id and etp."A"=parent_be.id and be.name ilike 'cd3 antigens';
select * from umls_graph where tail_id='C0108779' and relationship in ('isa', 'mapped_to', 'classified_as', 'has_mechanism_of_action', 'has_target', 'has_active_ingredient', 'tradename_of', 'has_phenotype');
"""
DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS = (
    ### GENERAL ###
    "isa",  # head/parent->tail/child, e.g. Meningeal Melanoma -> Adult Meningeal Melanoma
    "mapped_to",  # head-parent -> tail-child, e.g. Melanomas -> Uveal Melanoma
    "classified_as",  # head/parent -> tail/child, e.g. APPENDECTOMY -> Laparoscopic Appendectomy
    ### INTERVENTION ###
    "has_mechanism_of_action",  # head/MoA->tail/drug
    "has_target",  # head/target->tail/drug
    "has_active_ingredient",  # DROXIDOPA -> DROXIDOPA 100 mg ORAL CAPSULE
    "tradename_of",  # Amoxycillin -> Amoxil
    ### DISEASE ###
    "has_phenotype",  # head/disease->tail/phenotype, e.g. Mantle-Cell Lymphoma -> (specific MCL phenotype)
    ### INTERESTING BUT NOT FOR NOW ###
    # has_manifestation # e.g. Pains, Abdominal -> fabrys disease   (not suitable)
    # gene_associated_with_disease # e.g. Alzheimers Diseases -> PSEN1 Gene (interesting in future)
    # process_involves_gene # e.g. ABCB1 Gene -> Ligand Binding (questionable for this purpose)
    # biological_process_involves_gene_product / gene_product_plays_role_in_biological_process # questionable
    # has_doseformgroup # e.g. Injectables -> Buprenex Injectable Product (not helpful)
)
LEVEL_INSTANCE_THRESHOLD = 25
LEVEL_OVERRIDE_DELTA = 500
LEVEL_MIN_PREV_COUNT = 5
MAX_DEPTH_QUERY = 5
MAX_DEPTH_NETWORK = 6
DEFAULT_ANCESTOR_FILE = "data/umls_ancestors.json"


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(
        self,
        doc_type: DocType,
        instance_threshold: int,
        previous_threshold: int,
        current_override_threshold: int,
    ):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        # initialize superclass with _add_level_info transform
        super().__init__(transform_graph=self._add_level_info)
        self.doc_type = doc_type
        self.instance_threshold = instance_threshold
        self.previous_threshold = previous_threshold
        self.current_override_threshold = current_override_threshold

    @classmethod
    async def create(
        cls,
        filename: str | None = DEFAULT_ANCESTOR_FILE,
        doc_type: DocType = DocType.patent,
        instance_threshold: int = LEVEL_INSTANCE_THRESHOLD,
        previous_threshold: int = LEVEL_MIN_PREV_COUNT,
        current_override_threshold: int = LEVEL_OVERRIDE_DELTA,
    ) -> "AncestorUmlsGraph":
        """
        Factory for AncestorUmlsGraph

        Args:
            filename: filename to load from (if None, it will not load from nor save to a file)
            doc_type: doc type to use for graph
            instance_threshold: threshold for UMLS entry to be considered INSTANCE vs SUBINSTANCE
            previous_threshold: threshold for considering multiplier on prev count as criteria for level ++
            current_override_threshold: if the absolute change in counts between prev and current is greater than this, level ++
        """
        aug = cls(
            doc_type,
            instance_threshold=instance_threshold,
            previous_threshold=previous_threshold,
            current_override_threshold=current_override_threshold,
        )
        await aug.load(filename=filename)
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
            AND etu."A"=biomedical_entity.id
            AND umls.id=etu."B"
            AND umls.type_ids && $1
            GROUP BY umls.id

            UNION

            -- indication-mapped UMLS terms associated with docs, if preferred type
            SELECT umls.id as id, count(*) as count
            FROM biomedical_entity, indicatable, _entity_to_umls AS etu, umls
            WHERE biomedical_entity.id=indicatable.entity_id
            AND etu."A"=biomedical_entity.id
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
        - only {considered_relationships} or no relationship
        - limits types to biomedical
        - applies some naming restrictions (via 'suppressions')
        - suppresses entities whose name is also a type (indicates overly general)
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
                AND etu."A"=biomedical_entity.id
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
                AND etu."A"=biomedical_entity.id
                AND umls.id=etu."B"
                AND umls.type_ids && $1

                UNION

                -- UMLS to UMLS relationships
                -- e.g. C12345 & C67890
                SELECT
                    head_id as head,
                    tail_id as tail,
                    wt.depth + 1 as depth
                FROM umls_graph
                JOIN working_terms wt ON wt.head = tail_id
                JOIN umls as head_entity on head_entity.id = umls_graph.head_id
                JOIN umls as tail_entity on tail_entity.id = umls_graph.tail_id
                WHERE wt.depth <= {MAX_DEPTH_QUERY}
                AND relationship in {considered_relationships}
                AND head_entity.type_ids && $1
                AND tail_entity.type_ids && $1
                AND head_id not in {cui_suppressions}
                AND tail_id not in {cui_suppressions}
                AND head_id<>tail_id
                AND ts_lexize('english_stem', head_entity.type_names[1]) <> ts_lexize('english_stem', head_name)  -- exclude entities with a name that is also the type
                AND ts_lexize('english_stem', tail_entity.type_names[1]) <> ts_lexize('english_stem', tail_name)
                AND NOT head_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
                AND NOT tail_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
            )
            SELECT DISTINCT head, tail
            FROM working_terms
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, considered_tuis)
        logger.info("Edge query returned %s results", len(results))
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
        data = [v.get("count", 0) for v in list(g.nodes.values())]
        sns.displot(data, kde=True, aspect=10/4)
        ```
        """
        logger.info("Recursively propagating counts up the tree")

        def _propagate(g: DiGraph, node_id: str, depth: int = 0):
            child_ids: list[str] = list(g.successors(node_id))

            if depth < MAX_DEPTH_NETWORK:
                # for children with no counts, aka non-leaf nodes, recurse
                # e.g. if we're on Grandparent1, Parent1 and Parent2 have no counts,
                # so we recurse to set Parent1 and Parent2 counts based on their children
                for child_id in child_ids:
                    if g.nodes[child_id].get("count") is None:
                        _propagate(g, child_id, depth + 1)

            # set count to sum of all children counts
            g.nodes[node_id]["count"] = sum(
                g.nodes[child_id].get("count", 0) for child_id in child_ids
            )

        # take all nodes with no *incoming* edges, i.e. root nodes
        for node_id in [n for n, d in G.in_degree() if d == 0]:
            _propagate(G, node_id, 0)

        return G

    def _add_level_info(self, G: DiGraph) -> DiGraph:
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
                if current_count < self.instance_threshold:
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

            if current_prev_delta > self.current_override_threshold or (
                parent_current_delta > current_prev_delta
                and prev_count
                > self.previous_threshold  # avoid big changes in small numbers
            ):
                return increment_ontology_level(last_level)

            return OntologyLevel.NA

        def set_level(
            _G: DiGraph,
            node: NodeRecord,
            prev_node: NodeRecord | None = None,
            last_level: OntologyLevel | None = None,
            depth: int = 0,
        ) -> None:
            """
            Set level on node, and recurse through parents
            (mutation!)
            """
            # logger.info("Setting level, depth %s", depth)
            parent_ids = list(_G.predecessors(node.id))
            max_parent_count = (
                max([_G.nodes[p].get("count", 0) for p in parent_ids])
                if parent_ids
                else None
            )
            prev_count = prev_node.count if prev_node else None
            level = get_level(node, prev_count, last_level, max_parent_count)
            _G.nodes[node.id]["level"] = level

            # last real level as basis for inc
            new_last_level = level if level != OntologyLevel.NA else last_level

            # recurse through parents
            if depth < MAX_DEPTH_NETWORK:
                for parent_id in parent_ids:
                    parent = NodeRecord(**_G.nodes[parent_id])
                    # TODO: THIS MAY LOOP FOR A VERY LONG TIME
                    # (but doing a level=UNKNOWN check breaks logic)
                    set_level(
                        _G,
                        parent,
                        NodeRecord(**_G.nodes[node.id]),
                        new_last_level,
                        depth + 1,
                    )

        logger.info("Recursively propagating counts up the tree")

        # propogate counts up the tree
        new_g = AncestorUmlsGraph._propagate_counts(G.copy().to_directed())

        logger.info("Propagating levels down the tree")
        # take all nodes with no *outgoing* edges, i.e. leaf nodes
        for node_id in [n for n, d in new_g.out_degree() if d == 0]:
            node = new_g.nodes[node_id]
            if node.get("id") is None:
                logger.warning("Node %s has no id", node_id)
                node["id"] = node_id
            set_level(new_g, NodeRecord(**node))

        return new_g

    def get_ontology_level(self, cui: str) -> OntologyLevel:
        """
        Get ontology level for cui
        """
        if not cui in self.nodes:
            logger.warning("%s not in graph", cui)
            return OntologyLevel.NA

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

    def has_node(self, cui: str) -> bool:
        """
        See if cui is in graph
        """
        return cui in self.nodes
