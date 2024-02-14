import time
from typing import Sequence
import logging
from networkx import DiGraph
import networkx as nx
from prisma.enums import OntologyLevel
from pydash import flatten

from clients.low_level.prisma import prisma_client
from clients.umls.graph import UmlsGraph
from clients.umls.types import EdgeRecord, NodeRecord
from constants.umls import (
    PERMITTED_ANCESTOR_TYPES,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
)
from data.etl.entity.biomedical_entity.umls.types import compare_ontology_level
from typings.documents.common import DocType
from utils.classes import overrides
from utils.re import get_or_re

from .utils import increment_ontology_level


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
Debugging
select be.canonical_id, be.name, be.count, parent_be.name from biomedical_entity be, _entity_to_parent etp, biomedical_entity parent_be where etp."B"=be.id and etp."A"=parent_be.id and be.name ilike 'cd3 antigens';
select * from umls_graph where tail_id='C0108779' and relationship in ('isa', 'mapped_to', 'classified_as', 'has_mechanism_of_action', 'has_target', 'has_active_ingredient', 'tradename_of', 'has_phenotype');

select umls.count, umls.level, be.name, be2.name from biomedical_entity be, biomedical_entity be2, _entity_to_parent etp, umls  where etp."A"=be.id and etp."B"=be2.id and umls.id=be.canonical_id and be2.canonical_id='C0246631' order by umls.count desc;
select umls.id, umls.count, umls.level, umls.name ,  umlsp.id as parent_id, umlsp.count as parent_count, umlsp.level as parent_level, umlsp.name as parent_name from umls, umls umlsp, umls_graph ug where ug.head_id=umlsp.id and ug.tail_id=umls.id and relationship in ('isa', 'mapped_to', 'classified_as', 'has_mechanism_of_action', 'has_target', 'has_active_ingredient', 'tradename_of') and umls.id='C0246631' order by count desc limit 50;
"""
DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS = (
    ### GENERAL ###
    "isa",  # head/parent->tail/child, e.g. Meningeal Melanoma -> Adult Meningeal Melanoma
    "mapped_to",  # head-parent -> tail-child, e.g. Melanomas -> Uveal Melanoma
    "classified_as",  # head/parent -> tail/child, e.g. APPENDECTOMY -> Laparoscopic Appendectomy
    ### INTERVENTION ###
    "has_mechanism_of_action",  # head/MoA->tail/drug, e.g. Hormone Receptor Agonists [MoA] -> Lutropin Alpha
    "has_target",  # head/target->tail/drug
    "has_active_ingredient",  # DROXIDOPA -> DROXIDOPA 100 mg ORAL CAPSULE (important for clinical drugs)
    "tradename_of",  # Amoxycillin -> Amoxil; gatifloxacin 5 MG/ML Ophthalmic Solution -> gatifloxacin 5 MG/ML Ophthalmic Solution [Zymaxid]
    ### DISEASE ###
    "has_phenotype",  # head/disease->tail/phenotype, e.g. Mantle-Cell Lymphoma -> (specific MCL phenotype)
    ### INTERESTING BUT NOT FOR NOW ###
    # has_manifestation # e.g. Pains, Abdominal -> fabrys disease   (not suitable)
    # gene_associated_with_disease # e.g. Alzheimers Diseases -> PSEN1 Gene (interesting in future)
    # process_involves_gene # e.g. ABCB1 Gene -> Ligand Binding (questionable for this purpose)
    # biological_process_involves_gene_product / gene_product_plays_role_in_biological_process # questionable
    # has_doseformgroup # e.g. Injectables -> Buprenex Injectable Product (not helpful)
)
LEVEL_INSTANCE_THRESHOLD = 50
LEVEL_OVERRIDE_DELTA = 500
LEVEL_MIN_DELTA = 20
MAX_DEPTH_QUERY = 5
MAX_DEPTH_NETWORK = 15
DEFAULT_ANCESTOR_FILE = "data/umls_ancestors.json"
LEAF_KEY = "leaf"


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(
        self,
        doc_type: DocType,
        instance_threshold: int,
        delta_threshold: int,
        override_threshold: int,
        considered_tuis: list[str] = PERMITTED_ANCESTOR_TYPES,
    ):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        # initialize superclass with _add_hierarchy_info transform
        super().__init__(transform_graph=self._add_hierarchy_info)
        self.doc_type = doc_type
        self.instance_threshold = instance_threshold
        self.delta_threshold = delta_threshold
        self.override_threshold = override_threshold
        self.considered_tuis = considered_tuis

    @classmethod
    async def create(
        cls,
        filename: str | None = DEFAULT_ANCESTOR_FILE,
        doc_type: DocType = DocType.patent,
        instance_threshold: int = LEVEL_INSTANCE_THRESHOLD,
        delta_threshold: int = LEVEL_MIN_DELTA,
        override_threshold: int = LEVEL_OVERRIDE_DELTA,
    ) -> "AncestorUmlsGraph":
        """
        Factory for AncestorUmlsGraph

        Args:
            filename: filename to load from (if None, it will not load from nor save to a file)
            doc_type: doc type to use for graph
            instance_threshold: threshold for UMLS entry to be considered INSTANCE vs SUBINSTANCE
            delta_threshold: threshold for considering multiplier on prev count as criteria for level ++
            override_threshold: if the absolute change in counts between prev and current is greater than this, level ++
        """
        aug = cls(
            doc_type,
            instance_threshold=instance_threshold,
            delta_threshold=delta_threshold,
            override_threshold=override_threshold,
        )
        await aug.load(filename=filename)
        return aug

    @overrides(UmlsGraph)
    async def load_nodes(self) -> list[NodeRecord]:
        """
        Query nodes from umls
        """
        query = rf"""
            SELECT
                umls.id AS id,
                umls.name AS name,
                COALESCE(SUM(docs.count), 0) AS count,
                CASE
                    -- T200 / Clinical Drug is always SUBINSTANCE
                    WHEN array_length(umls.type_ids, 1)=1 AND umls.type_ids[1]='T200'
                        THEN 'SUBINSTANCE'
                    ELSE null
                    END AS level_override,
                umls.type_ids AS type_ids
            FROM umls, _entity_to_umls AS etu
            LEFT JOIN (
                SELECT entity_id, sum(count) AS count
                FROM (
                    SELECT entity_id, count(*) AS count
                    FROM intervenable
                    GROUP BY entity_id

                    UNION

                    SELECT entity_id, count(*) AS count
                    FROM indicatable
                    GROUP BY entity_id
                ) int GROUP BY entity_id
            ) docs ON docs.entity_id=etu."A"
            WHERE umls.id=etu."B"
            GROUP BY umls.id

            UNION

            SELECT
                id,
                name,
                0 AS count,
                CASE
                    -- T200 / Clinical Drug is always SUBINSTANCE
                    WHEN array_length(umls.type_ids, 1)=1 AND umls.type_ids[1]='T200'
                    THEN 'SUBINSTANCE'
                    ELSE null
                    END AS level_override,
                umls.type_ids AS type_ids
            FROM umls
            WHERE id NOT IN (SELECT "B" FROM _entity_to_umls where "B"=id)
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, self.considered_tuis)
        return [NodeRecord(**r) for r in results]

    @overrides(UmlsGraph)
    async def load_edges(
        self,
        considered_relationships: tuple[str, ...] = DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS,
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
                    distinct umls.id AS head,
                    '{LEAF_KEY}' AS tail,
                    1 AS depth
                FROM umls, _entity_to_umls AS etu
                LEFT JOIN (
                    SELECT distinct entity_id
                    FROM intervenable

                    UNION

                    SELECT distinct entity_id
                    FROM indicatable
                ) docs ON docs.entity_id=etu."A"
                WHERE umls.id=etu."B"

                UNION

                -- UMLS to UMLS relationships
                -- e.g. C12345 & C67890
                SELECT
                    head_id AS head,
                    tail_id AS tail,
                    wt.depth + 1 AS depth
                FROM umls_graph
                JOIN working_terms wt ON wt.head = tail_id
                JOIN umls AS head_entity ON head_entity.id = umls_graph.head_id
                JOIN umls AS tail_entity ON tail_entity.id = umls_graph.tail_id
                WHERE wt.depth <= {MAX_DEPTH_QUERY}
                AND head_entity.type_ids && $1
                AND tail_entity.type_ids && $1
                AND relationship IN {considered_relationships}
                AND head_id NOT IN {cui_suppressions}
                AND tail_id NOT IN {cui_suppressions}
                AND head_id<>tail_id
                AND ts_lexize('english_stem', head_entity.type_names[1]) <> ts_lexize('english_stem', head_name)  -- exclude entities with a name that is also the type
                AND ts_lexize('english_stem', tail_entity.type_names[1]) <> ts_lexize('english_stem', tail_name)
                AND NOT head_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
                AND NOT tail_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
            )
            SELECT DISTINCT head, tail
            FROM working_terms
            """

        print(query)

        client = await prisma_client(300)
        results = await client.query_raw(query, self.considered_tuis)

        # omit source edges (cui -> LEAF_KEY ("leaf"))
        edges = [EdgeRecord(**r) for r in results if r["tail"] != LEAF_KEY]
        logger.info("Returning %s edges", len(edges))
        return edges

    @staticmethod
    def _set_counts(G: DiGraph) -> DiGraph:
        """
        Sets counts recursively up the tree
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
        logger.info("Recursively setting counts")

        def collect_counts(
            g: DiGraph, node_id: str, depth: int = 0
        ) -> list[tuple[str, int]]:
            # count all descendants (children, grandchildren, etc)
            # to avoiding the double counting issue occurring with a cascade approach
            descendent_ids = nx.descendants(g, node_id)

            # in case parent has its own count, i.e. biomedical ents link directly to it.
            existing_count = int(g.nodes[node_id].get("count") or 0)
            count = (
                sum(
                    g.nodes[descendent_id].get("count") or 0
                    for descendent_id in descendent_ids
                )
                + existing_count
            )

            if depth < MAX_DEPTH_NETWORK:
                child_counts = flatten(
                    [
                        collect_counts(g, child_id, depth + 1)
                        for child_id in g.successors(node_id)
                    ]
                )
            else:
                child_counts = []

            return [(node_id, count)] + child_counts

        # take all nodes with no *incoming* edges, i.e. root nodes
        count_map = dict(
            flatten(
                [
                    collect_counts(G, node_id, 0)
                    for node_id in [n for n, d in G.in_degree() if d == 0]
                ]
            )
        )
        nx.set_node_attributes(G, count_map, "count")

        return G

    def _get_level(
        self,
        current_node: NodeRecord,
        prev_count: int,
        last_level: OntologyLevel,
        max_parent_count: int | None = None,
    ) -> OntologyLevel:
        """
        Determine level of current node
        """

        # i.e. for Clinical Drug / T200 -> SUBINSTANCE
        if current_node.level_override is not None:
            return current_node.level_override

        current_count = current_node.count or 0

        if prev_count == 0 or last_level == OntologyLevel.UNKNOWN:
            """
            leaf node, or effective leaf node (i.e. previous had no count)
            """
            if current_count < self.instance_threshold:
                return OntologyLevel.SUBINSTANCE

            return OntologyLevel.INSTANCE

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

        if current_prev_delta > self.override_threshold or (
            parent_current_delta > current_prev_delta
            and current_prev_delta > self.delta_threshold
        ):
            return increment_ontology_level(last_level)

        return OntologyLevel.NA

    def _set_level(
        self,
        G: DiGraph,
        node: NodeRecord,
        prev_node: NodeRecord | None = None,
        last_level: OntologyLevel = OntologyLevel.UNKNOWN,
        depth: int = 0,
    ) -> None:
        """
        Set level on node, and recurse through parents

        Note: Uses mutation / is recursive

        Args:
            G (DiGraph): graph
            node (NodeRecord): node to set level on
            prev_node (NodeRecord, Optional): previous node
            last_level (OntologyLevel): last level
            depth: int: depth in the tree
        """

        prev_count = prev_node.count if prev_node else 0

        parent_ids = list(G.predecessors(node.id))
        max_parent_count: int | None = (
            max([G.nodes[p].get("count", 0) for p in parent_ids])
            if parent_ids
            else None
        )

        level = self._get_level(node, prev_count or 0, last_level, max_parent_count)

        # level may have already been set via a different path
        existing_level = G.nodes[node.id].get("level") or OntologyLevel.UNKNOWN

        # if level is lower (e.g. L1_CATEGORY) than existing level (e.g. L2_CATEGORY), keep existing level
        if compare_ontology_level(level, existing_level) <= 0:
            level = existing_level

        G.nodes[node.id]["level"] = level
        logger.debug(
            "Setting level for %s from %s to %s (existing %s)",
            node.id,
            last_level,
            level,
            existing_level,
        )

        # last non-NA level as basis for increments
        # if last_level == L1_CATEGORY, level == NA, last_level == L1_CATEGORY
        # so level can be set to L2_CATEGORY in future iterations
        new_last_level = level if level != OntologyLevel.NA else last_level

        # recurse through parents
        if depth < MAX_DEPTH_NETWORK:
            for parent_id in parent_ids:
                parent = NodeRecord(**G.nodes[parent_id])
                self._set_level(
                    G,
                    parent,
                    NodeRecord(**G.nodes[node.id]),
                    new_last_level,
                    depth + 1,
                )

    def _add_hierarchy_info(self, G: DiGraph) -> DiGraph:
        """
        Add ontology level to nodes

        Calls _propagate_counts to set counts on all nodes first.
        """

        # set counts on all nodes
        new_g = AncestorUmlsGraph._set_counts(G.copy().to_directed())

        logger.info("Propagating levels up the tree")
        # take all nodes with no *outgoing* edges, i.e. leaf nodes
        for node_id in [n for n, d in new_g.out_degree() if d == 0]:
            node = NodeRecord(**new_g.nodes[node_id])
            self._set_level(new_g, node)

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
