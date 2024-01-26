from abc import abstractmethod
import math
import time
import networkx as nx
import logging

from prisma.enums import OntologyLevel

from data.etl.entity.biomedical_entity.umls.types import get_next_ontology_level


from .types import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INSTANCE_THRESHOLD = 25


class UmlsGraph(object):
    """
    Abstract class for UMLS graph
    Computes betweenness centrality for nodes, which is used for ancestor selection.

    TODO: no reason this needs to be UMLS-specific
    """

    def __init__(self):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        pass

    async def load(self):
        self.G = await self.load_graph()
        self.nodes: dict[str, dict] = dict(self.G.nodes.data())

    @abstractmethod
    async def get_edges(self) -> list[EdgeRecord]:
        """
        Query edges from umls
        """
        raise NotImplementedError

    @abstractmethod
    async def get_nodes(self) -> list[NodeRecord]:
        """
        Query nodes from umls
        """
        raise NotImplementedError

    async def load_graph(self) -> nx.DiGraph:
        """
        Load UMLS graph from database
        """
        start = time.monotonic()
        logger.info("Loading graph")
        G = nx.DiGraph()

        nodes = await self.get_nodes()
        edges = await self.get_edges()

        if len(nodes) == 0:
            raise ValueError("No nodes found")

        if len(edges) == 0:
            raise ValueError("No edges found")

        G.add_nodes_from([(n.id, n) for n in nodes])
        G.add_edges_from([(e.head, e.tail) for e in edges])

        # add levels
        G = self._add_level_info(G)

        logger.info(
            "Graph has %s nodes, %s edges (took %s seconds)",
            G.number_of_nodes(),
            G.number_of_edges(),
            round(time.monotonic() - start),
        )
        return G

    @staticmethod
    def _propagate_counts(G: nx.DiGraph) -> nx.DiGraph:
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

        def _propagate(g: nx.DiGraph, node: NodeRecord):
            children = g.successors(node)

            # for children with no counts, aka non-leaf nodes, recurse
            # e.g. if we're on Grandparent1, Parent1 and Parent2 have no counts,
            # so we recurse to set Parent1 and Parent2 counts based on their children
            for child in children:
                if g.nodes[child].get("count") is None:
                    _propagate(g, child)

            # set count to sum of all children counts
            g.nodes[node]["count"] = sum(g.nodes[child]["count"] for child in children)

        # take all nodes with no *incoming* edges, i.e. root nodes
        for node in [n for n, d in G.in_degree() if d == 0]:
            _propagate(G, node)

        return G

    @staticmethod
    def _add_level_info(G: nx.DiGraph) -> nx.DiGraph:
        """
        Add ontology level to nodes

        Calls _propagate_counts to set counts on all nodes first.
        """
        logger.info("Recursively propagating counts up the tree")

        def set_level(_G: nx.DiGraph, node, prev_node: dict | None = None):
            parents = _G.predecessors(node)
            max_parent_count = (
                max([_G.nodes[p]["count"] for p in parents]) if parents else None
            )
            prev_count = prev_node["count"] if prev_node else 0
            current_count = node["count"]

            if prev_node is None:
                """
                leaf node. level it INSTANCE if sufficiently common.
                """
                if node["count"] < INSTANCE_THRESHOLD:
                    level = OntologyLevel.SUBINSTANCE
                else:
                    level = OntologyLevel.INSTANCE
            elif max_parent_count is None:
                """
                root node. level it one higher than prev node.
                """
                level = get_next_ontology_level(prev_node["level"])
            else:
                """
                compare rate of change (in count / cumulative docs) between:
                    - prev node and current node
                    - current node and max parent node
                """

                dcdprev = current_count - prev_count
                dparentdc = max_parent_count - current_count

                # e.g. 100 150 175 -> 50, 25 -> 0.5 ... slow, plateauing
                # so it isn't that meaningful to delineate this (vs the next) level
                if dparentdc / dcdprev < 2:
                    level = OntologyLevel.NA
                # e.g. 5 50 1000 -> 45, 950 -> 21 ... fast growth
                # so it's meaningful to delineate this level
                else:
                    level = get_next_ontology_level(prev_node["level"])

            _G.nodes[node]["level"] = level

            for parent in parents:
                set_level(_G, parent, node)

        # propogate counts up the tree
        new_g = UmlsGraph._propagate_counts(G.copy().to_directed())

        # take all nodes with no *outgoing* edges, i.e. leaf nodes
        for node in [n for n, d in new_g.out_degree() if d == 0]:
            set_level(new_g, node)

        return new_g
