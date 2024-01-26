from abc import abstractmethod
import time
from typing import Callable
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

    def __init__(
        self, transform_graph: Callable[[nx.DiGraph], nx.DiGraph] | None = None
    ):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        self.transform_graph = transform_graph

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

        if self.transform_graph:
            G = self.transform_graph(G)

        logger.info(
            "Graph has %s nodes, %s edges (creation took %s seconds)",
            G.number_of_nodes(),
            G.number_of_edges(),
            round(time.monotonic() - start),
        )
        return G
