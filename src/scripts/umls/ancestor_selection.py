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
            " ".join([rf"and not {table}.str ~ '\y{s}\y'" for s in suppressions])
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

        head_name_sql = UmlsGraph._format_name_sql("head_entity", suppressions)
        tail_name_sql = UmlsGraph._format_name_sql("tail_entity", suppressions)
        name_sql = head_name_sql + "\n" + tail_name_sql

        ancestory_edges = self.db.select(
            f"""
            SELECT cui1 as head, cui2 as tail
            FROM mrrel as relationship,
            mrhier as hierarchy,
            mrsty as head_semantic_type,
            mrsty as tail_semantic_type,
            mrconso as head_entity,
            mrconso as tail_entity
            where head_entity.cui = cui1
            and tail_entity.cui = cui2
            and relationship.cui1 = head_semantic_type.cui
            and relationship.cui2 = tail_semantic_type.cui
            and hierarchy.cui = relationship.cui1
            and hierarchy.ptr is not null                                 -- suppress entities wo parent (otherwise overly general)
            and relationship.rel in ('RN', 'CHD')                         -- narrower, child
            and (relationship.rela is null or relationship.rela = 'isa')  -- no specified relationship, or 'is a'
            and head_semantic_type.tui in {BIOMEDICAL_UMLS_TYPES}
            and tail_semantic_type.tui in {BIOMEDICAL_UMLS_TYPES}
            and ts_lexize('english_stem', head_semantic_type.sty) <> ts_lexize('english_stem', head_entity.str)   -- exclude entities with a name that is also the type (indicates an overly general)
            and ts_lexize('english_stem', tail_semantic_type.sty) <> ts_lexize('english_stem', tail_entity.str)
            {name_sql}                                                                                            -- applies lang and name filters
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

    def _get_betweenness_subgraph(self, k: int = 50000):
        """
        Get subgraph of top k nodes by degree, excluding "hubs" (high degree nodes)

        Args:
            k: number of nodes to include in the map (for performance reasons)
            hub_degree_threshold: nodes with degree above this threshold will be excluded
        """
        degrees: list[tuple[str, int]] = self.G.degree  # type: ignore
        top_nodes = [
            node for (node, _) in sorted(degrees, key=lambda x: x[1], reverse=True)[:k]
        ]
        return self.G.subgraph(top_nodes)

    def load_betweenness(
        self,
        k: int = 50000,
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
        bc_subgraph = self._get_betweenness_subgraph(k)

        logger.info("Loading betweenness centrality map (slow)...")
        bc_map = betweenness_centrality_parallel(bc_subgraph)

        if file_name is not None:
            logger.info("Saving bc map to %s", file_name)
            save_json_as_file(bc_map, file_name)
        else:
            logger.warning("Not saving bc map to file, because no filename specified.")

        logger.info("Loaded bc map in %s seconds", round(time.monotonic() - start))
        return bc_map
