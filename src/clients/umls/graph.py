from abc import abstractmethod
import time
from typing import Callable
import networkx as nx
import logging
from pydash import uniq

from utils.file import maybe_load_pickle, save_as_pickle

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

    async def load(self, filename: str | None = None):
        g = None
        if filename is not None:
            g = maybe_load_pickle(filename)
            logger.info("Loaded graph from %s: %s", filename, g is not None)

        if g is None:
            logger.info("Loading graph from database")
            g = await self.load_graph()
            if filename is not None:
                save_as_pickle(g, filename)

        self.G = g
        self.nodes: dict[str, dict] = dict(g.nodes.data())

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
        G = nx.DiGraph()

        nodes = await self.get_nodes()
        edges = await self.get_edges()

        if len(nodes) == 0:
            raise ValueError("No nodes found")

        if len(edges) == 0:
            raise ValueError("No edges found")

        # add nodes
        G.add_nodes_from([(n.id, n) for n in nodes])

        # temp hack: add nodes for edge ids that don't have nodes
        node_ids = [n.id for n in nodes]
        edge_node_ids = uniq([e.head for e in edges] + [e.tail for e in edges])
        new_nodes = [id for id in edge_node_ids if id not in node_ids]
        logger.info("Adding %s nodes from edges", len(new_nodes))
        G.add_nodes_from([(id, {"id": id}) for id in new_nodes])

        # add edges
        G.add_edges_from([(e.head, e.tail) for e in edges])

        if nx.is_directed_acyclic_graph(G) == False:
            cycles = nx.find_cycle(G)
            logger.error("Graph is cyclic: %s", cycles[0:10])
            G.remove_edges_from(cycles)

        if self.transform_graph:
            G = self.transform_graph(G)

        logger.info(
            "Graph has %s nodes, %s edges (creation took %s seconds)",
            G.number_of_nodes(),
            G.number_of_edges(),
            round(time.monotonic() - start),
        )
        return G
