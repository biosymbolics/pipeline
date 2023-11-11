from multiprocessing import Pool
import time
import networkx as nx
import logging
from pydash import is_list

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from utils.file import load_json_from_file, save_json_as_file
from utils.graph import betweenness_centrality_parallel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HUB_DEGREE_THRESHOLD = 300
BETWEENNESS_FILE = "umls_betweenness.json"

# could include things like virus, bacteria, etc.
BIOMEDICAL_TYPES = tuple(
    [
        # "T023"  # Body Part, Organ, or Organ Component
        "T020",  # acquired abnormality
        "T019",  # congenital abnormality
        # "T025",  # cell
        # "T026",  # cell component
        "T043",  # cell function
        "T028",  # "Gene or Genome",
        # "T033", # Finding
        # "T034",  # laboratory or test result
        "T037",  # Injury or Poisoning
        "T044",  # molecular function
        "T046",  # Pathologic Function
        "T047",  # "Disease or Syndrome",
        "T048",  # "Mental or Behavioral Dysfunction",
        "T049",  # "Cell or Molecular Dysfunction",
        "T046",  # "Pathologic Function",
        "T059",  # laboratory procedure
        "T060",  # diagnostic procedure
        "T061",  # "Therapeutic or Preventive Procedure",
        "T063",  # research activity
        "T074",  # medical device
        "T075",  # research device
        "T085",  # "Molecular Sequence",
        "T086",  # "Nucleotide Sequence",
        "T087",  # "Amino Acid Sequence",
        "T088",  # "Carbohydrate Sequence",
        "T103",  # "Chemical",
        "T104",  # "Chemical Viewed Structurally",
        "T109",  # "Organic Chemical",
        "T114",  # "Nucleic Acid, Nucleoside, or Nucleotide",
        "T116",  # "Amino Acid, Peptide, or Protein",
        "T120",  # "Chemical Viewed Functionally",
        "T121",  # "Pharmacologic Substance",
        "T122",  # biomedical or dental material
        "T123",  # "Biologically Active Substance",
        "T125",  # "Hormone",
        "T129",  # "Immunologic Factor",
        "T126",  # "Enzyme",
        "T127",  # "Vitamin",
        "T129",  # "Immunologic Factor",
        "T130",  # "Indicator, Reagent, or Diagnostic Aid",
        "T131",  # "Hazardous or Poisonous Substance",
        "T167",  # "Substance",
        "T168",  # food
        "T184",  # sign or symptom
        "T190",  # Anotomic abnormality
        "T191",  # "Neoplastic Process",
        "T192",  # "Receptor",
        "T195",  # antibiotic
        "T196",  # "Element, Ion, or Isotope"
        "T197",  # "Inorganic Chemical"
        "T200",  # "Clinical Drug"
        "T203",  # drug delivery device
        # "T201",  # Clinical Attribute
    ]
)


class UmlsGraph:
    """
    UMLS graph in NetworkX form

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

    def load_graph(self) -> nx.Graph:
        """
        Load UMLS graph from database

        Restricted to ancestoral relationships between biomedical entities
        """
        start = time.monotonic()
        logger.info("Loading UMLS into graph")
        G = nx.Graph()
        ancestory_edges = self.db.select(
            f"""
            SELECT cui1 as head, cui2 as tail
            FROM mrrel, mrhier, mrsty
            where mrhier.cui = mrrel.cui1
            and mrrel.cui1 = mrsty.cui
            and mrhier.ptr is not null -- only including entities that have a parent
            and mrrel.rel in ('RN', 'CHD') -- narrower, child
            and (mrrel.rela is null or mrrel.rela = 'isa') -- no specified relationship, or 'is a'
            and mrsty.tui in {BIOMEDICAL_TYPES}
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
        """
        if file_name is not None:
            try:
                return load_json_from_file(file_name)
            except FileNotFoundError:
                pass

        start = time.monotonic()
        bc_subgraph = self._get_betweenness_subgraph(k, hub_degree_threshold)

        logger.info("Loading betweenness centrality map (slow)...")
        bet_cen = betweenness_centrality_parallel(bc_subgraph)
        save_json_as_file(bet_cen, BETWEENNESS_FILE)

        logger.info("Loaded bc map in %s seconds", round(time.monotonic() - start))
        return bet_cen
