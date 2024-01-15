from abc import abstractmethod
import time
from typing import Sequence
import networkx as nx
import logging

from clients.low_level.prisma import prisma_client
from constants.umls import UMLS_NAME_SUPPRESSIONS
from utils.file import load_json_from_file, save_json_as_file
from utils.graph import betweenness_centrality_parallel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BETWEENNESS_FILE = "umls_betweenness.json"


class UmlsGraph(object):
    """
    Abstract class for UMLS graph
    Computes betweenness centrality for nodes, which is used for ancestor selection.
    """

    def __init__(self, file_name: str = BETWEENNESS_FILE):
        """
        ***Either use a factory method in the subclass, or call load() after init***
        """
        self.bc_file = file_name

    async def load(self):
        self.G = await self.load_graph()
        self.nodes: dict[str, dict] = dict(self.G.nodes.data())
        try:
            self.betweenness_map = load_json_from_file(self.bc_file)
        except FileNotFoundError:
            self.betweenness_map = self._load_betweenness()

    @abstractmethod
    def edge_query(self, suppressions: Sequence[str] | set[str]) -> str:
        """
        Query edges from umls

        TODO: make "get_edges" and enforce type (head/tail)
        """
        raise NotImplementedError

    async def load_graph(
        self, suppressions: Sequence[str] | set[str] = UMLS_NAME_SUPPRESSIONS
    ) -> nx.Graph:
        """
        Load UMLS graph from database

        Restricted to ancestoral relationships between biomedical entities
        """
        start = time.monotonic()
        logger.info("Loading UMLS into graph")
        G = nx.Graph()

        client = await prisma_client(300)
        edges = await client.query_raw(self.edge_query(suppressions))
        G.add_edges_from([(e["head"], e["tail"]) for e in edges])

        logger.info(
            "Graph has %s nodes, %s edges (took %s seconds)",
            G.number_of_nodes(),
            G.number_of_edges(),
            round(time.monotonic() - start),
        )
        return G

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

    def _load_betweenness(
        self,
        k: int = 10000,
        file_name: str | None = BETWEENNESS_FILE,
    ) -> dict[str, float]:
        """
        Load betweenness centrality map

        Args:
            k: number of nodes to include in the map (for performance reasons)
            hub_degree_threshold: nodes with degree above this threshold will be excluded
            file_name: if provided, will load from file instead of computing (or save to file after computing)

        11/23 - Takes roughly 1 hour for 50k nodes using 6 cores
        1/10 - Took roughly 3 hours (maybe?) for 10k nodes using 6 cores
        """
        start = time.monotonic()
        bc_subgraph = self._get_topk_subgraph(k)

        sample_k = round(k / 4)
        logger.info("Calcing betweenness centrality (top %s; sampling %s)", k, sample_k)
        bc_map = betweenness_centrality_parallel(bc_subgraph, sample_k)

        if file_name is not None:
            logger.info("Saving bc map to %s", file_name)
            save_json_as_file(bc_map, file_name)
        else:
            logger.warning("Not saving bc map to file, because no filename specified.")

        logger.info("Loaded bc map in %s seconds", round(time.monotonic() - start))
        return bc_map
