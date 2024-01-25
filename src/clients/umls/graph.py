from abc import abstractmethod
import time
import networkx as nx
import logging


from .types import EdgeRecord, NodeRecord

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

        # propogate counts up the tree
        G = self._propagate_counts(G)

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
        """
        logger.info("Recursively propagating counts up the tree")

        def _execute(_G, node):
            children = _G.successors(node)
            node_weight = sum(_G.nodes[child]["count"] for child in children)
            _G.nodes[node]["count"] = node_weight

            for child in children:
                _execute(_G, child)

        new_g = G.copy().to_directed()
        for node in new_g.nodes():
            # start from leaf nodes
            if new_g.out_degree(node) == 0:
                _execute(new_g, node)

        return new_g

    def _get_topk_subgraph(self, k: int, max_degree: int = 1000):
        """
        Get subgraph of top k nodes by degree

        Args:
            k: number of nodes to include in the map (for performance reasons)
        """
        nodes: list[tuple[str, int]] = self.G.degree  # type: ignore

        # non-trivial hubs (i.e. not massive and therefore meaningless)
        nontrivial_hubs = [
            (node, degree)
            for (node, degree) in nodes
            if degree < max_degree and degree > 0
        ]

        # top k nodes by degree
        top_nodes = sorted(nontrivial_hubs, key=lambda x: x[1], reverse=True)[:k]

        return self.G.subgraph([node for (node, _) in top_nodes])
