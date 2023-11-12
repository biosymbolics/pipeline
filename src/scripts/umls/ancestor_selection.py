import time
from typing import Sequence
import networkx as nx
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from data.common.biomedical.constants import (
    BIOMEDICAL_UMLS_TYPES,
    UMLS_NAME_SUPPRESSIONS,
)
from utils.file import load_json_from_file, save_json_as_file
from utils.graph import betweenness_centrality_parallel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HUB_DEGREE_THRESHOLD = 300
BETWEENNESS_FILE = "umls_betweenness.json"


class UmlsGraph:
    """
    UMLS graph in NetworkX form
    Most importantly, computes betweenness centrality for nodes, which is used for ancestor selection.

    Usage:
    ```
    import logging; import sys;
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    from scripts.umls.ancestor_selection import UmlsGraph;
    g = UmlsGraph()
    ```
    """

    def __init__(self):
        umls_db = f"{BASE_DATABASE_URL}/umls"
        self.db = PsqlDatabaseClient(umls_db)
        self.G = self.load_graph()
        self.betweenness_map = self.load_betweenness()

    @staticmethod
    def _format_name_sql(table: str, suppressions: Sequence[str] | set[str]) -> str:
        name_filter = (
            " ".join([f"and not {table}.str ~ '^.*{s}'" for s in suppressions])
            if len(suppressions) > 0
            else ""
        )
        lang_filter = f"""
            and {table}.lat='ENG' -- english
            and {table}.ts='P' -- preferred terms
            and {table}.ispref='Y' -- preferred term
        """
        return name_filter + lang_filter

    def load_graph(
        self, suppressions: Sequence[str] | set[str] = UMLS_NAME_SUPPRESSIONS
    ) -> nx.Graph:
        """
        Load UMLS graph from database

        Restricted to ancestoral relationships between biomedical entities

        NOTE: assumes suppressions are "ends-with" strings and proper casing, for perf.
        """
        start = time.monotonic()
        logger.info("Loading UMLS into graph")
        G = nx.Graph()

        head_name_sql = UmlsGraph._format_name_sql("head_entities", suppressions)
        tail_name_sql = UmlsGraph._format_name_sql("tail_entities", suppressions)

        ancestory_edges = self.db.select(
            f"""
            SELECT cui1 as head, cui2 as tail
            FROM mrrel, mrhier, mrsty, mrconso head_entities, mrconso tail_entities
            where mrhier.cui = mrrel.cui1
            and mrrel.cui1 = mrsty.cui
            and head_entities.cui = cui1
            and tail_entities.cui = cui2
            and mrhier.ptr is not null -- only including entities that have a parent
            and mrrel.rel in ('RN', 'CHD') -- narrower, child
            and (mrrel.rela is null or mrrel.rela = 'isa') -- no specified relationship, or 'is a'
            and mrsty.tui in {BIOMEDICAL_UMLS_TYPES}
            {head_name_sql}
            {tail_name_sql}
            group by cui1, cui2
            """
        )

        G.add_edges_from([(e["head"], e["tail"]) for e in ancestory_edges])

        logger.info(
            "Graph has %s nodes, %s edges (took %s seconds)",
            G.number_of_nodes(),
            G.number_of_edges(),
            round(time.monotonic() - start),
        )
        return G

    def _get_betweenness_subgraph(
        self, k: int = 50000, hub_degree_threshold: int = HUB_DEGREE_THRESHOLD
    ):
        """
        Get subgraph of top k nodes by degree, excluding "hubs" (high degree nodes)

        Args:
            k: number of nodes to include in the map (for performance reasons)
            hub_degree_threshold: nodes with degree above this threshold will be excluded
        """
        degrees: list[tuple[str, int]] = self.G.degree  # type: ignore
        non_hub_degrees = [d for d in degrees if d[1] < hub_degree_threshold]
        top_nodes = [
            node
            for (node, _) in sorted(non_hub_degrees, key=lambda x: x[1], reverse=True)[
                :k
            ]
        ]
        return self.G.subgraph(top_nodes)

    def load_betweenness(
        self,
        k: int = 50000,
        hub_degree_threshold: int = HUB_DEGREE_THRESHOLD,
        file_name: str | None = BETWEENNESS_FILE,
    ) -> dict[str, float]:
        """
        Load betweenness centrality map

        Args:
            k: number of nodes to include in the map (for performance reasons)
            hub_degree_threshold: nodes with degree above this threshold will be excluded
            file_name: if provided, will load from file instead of computing (or save to file after computing)

        11/23 - Takes roughly 1 hour for 50k nodes using 6 cores
        """
        if file_name is not None:
            try:
                return load_json_from_file(file_name)
            except FileNotFoundError:
                pass

        start = time.monotonic()
        bc_subgraph = self._get_betweenness_subgraph(k, hub_degree_threshold)

        logger.info("Loading betweenness centrality map (slow)...")
        bc_map = betweenness_centrality_parallel(bc_subgraph)

        if file_name is not None:
            logger.info("Saving bc map to %s", file_name)
            save_json_as_file(bc_map, file_name)
        else:
            logger.warning("Not saving bc map to file, because no filename specified.")

        logger.info("Loaded bc map in %s seconds", round(time.monotonic() - start))
        return bc_map
