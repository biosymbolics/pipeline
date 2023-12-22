"""
Patent graph reports
"""

from typing import Sequence
import logging
import networkx as nx
from pydash import uniq

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from clients.patents.constants import ENTITY_DOMAINS
from constants.core import ANNOTATIONS_TABLE, APPLICATIONS_TABLE, TERM_IDS_TABLE
from typings.patents import PatentApplication

from .types import Node, Link, SerializableGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NODES = 100
MIN_NODE_DEGREE = 2

RELATIONSHIPS_OF_INTEREST = [
    "allelic_variant_of",
    "biological_process_involves_gene_product",
    # "chemical_or_drug_has_mechanism_of_action",
    # "chemical_or_drug_has_physiologic_effect",
    # "genetic_biomarker_related_to",
    "gene_involved_in_molecular_abnormality",
    "gene_plays_role_in_process",
    "gene_product_encoded_by_gene",
    "gene_product_has_biochemical_function",
    "gene_encodes_gene_product",
    "gene_is_element_in_pathway",
    "gene_product_plays_role_in_process",
    "gene_involved_in_pathogenesis_of_disease",
    "gene_product_variant_of_gene_product",
    "gene_product_has_gene_product_variant",
    "has_allelic_variant",
    "has_phenotype",
    # "has_manifestation",
    "has_gene_product_element",
    # "has_therapeutic_class",
    # "has_mechanism_of_action",
    # "has_physiologic_effect",
    "has_target",
    # "is_mechanism_of_action_of_chemical_or_drug",
    # "is_physiologic_effect_of_chemical_or_drug",
    "is_target",
    # "is_target_of",
    # "may_treat",
    # "may_be_treated_by",
    # "manifestation_of",
    # "mechanism_of_action_of",
    "molecular_abnormality_involves_gene",
    "negatively_regulates",
    "negatively_regulated_by",
    "pathogenesis_of_disease_involves_gene",
    "pathway_has_gene_element",
    # "phenotype_of",
    # "physiologic_effect_of",
    "positively_regulates",
    "positively_regulated_by",
    "process_involves_gene",
    # "related_to_genetic_biomarker",
    "regulated_by",
    "regulates",
    # "therapeutic_class_of",
]

ENTITY_GROUP = "entity"
PATENT_GROUP = "patent"


def generate_graph(
    relationships: Sequence[dict], max_nodes: int = MAX_NODES
) -> nx.Graph:
    """
    Take UMLS query results and turn into a graph
    """
    g = nx.Graph()
    g.add_edges_from(
        [(r["head"], r["tail"]) for r in relationships if r["head"] != r["tail"]]
    )
    node_to_group = {
        **{r["head"]: r["group"] for r in relationships},
        **{r["tail"]: ENTITY_GROUP for r in relationships},
    }
    nx.set_node_attributes(g, node_to_group, "group")

    if isinstance(g.degree, int):
        raise Exception("Graph has no nodes")

    degree_map = {n: d for (n, d) in g.degree}
    nx.set_node_attributes(g, degree_map, "size")

    weights = {
        (r["head"], r["tail"]): min(degree_map[r["tail"]] / 4, 20)
        for r in relationships
    }
    nx.set_edge_attributes(g, values=weights, name="weight")

    top_nodes = [
        node
        for (node, _) in sorted(
            [(n, d) for (n, d) in g.degree if d >= MIN_NODE_DEGREE],
            key=lambda x: x[1],
            reverse=True,
        )[:max_nodes]
    ]
    subgraph = g.subgraph(
        uniq(
            [
                *[n[0] for n in list(g.nodes(data="group")) if n[1] == PATENT_GROUP],  # type: ignore
                *top_nodes,
            ]
        )
    )

    return subgraph


def graph_patent_relationships(
    patents: Sequence[PatentApplication],
    max_nodes: int = MAX_NODES,
) -> SerializableGraph:
    """
    Graph UMLS ancestory for a set of patents

    - Nodes are UMLS entities
    - Edges are ancestory relationships OR patent co-occurrences (e.g. PD and anti-alpha-synucleins)
    - Nodes contain metadata of the relevant patents

    Note: perhaps inefficient to load patents just for ids, but patent call is cached (+ used by other queries)
    """
    patent_ids = [p["publication_number"] for p in patents]

    # TODO: save relationship type, ancestors
    sql = f"""
        -- patent-node to entity relationships
        SELECT
            TO_CHAR(app.priority_date, 'YYYY') as head,
            umls.canonical_name as tail,
            count(*) as weight,
            '{PATENT_GROUP}' as group
        FROM
            {ANNOTATIONS_TABLE} a,
            {APPLICATIONS_TABLE} app,
            {TERM_IDS_TABLE} t,
            umls_lookup umls
        WHERE a.publication_number = ANY(ARRAY{patent_ids})
        AND app.publication_number = a.publication_number
        AND a.domain in {tuple(ENTITY_DOMAINS)}
        AND t.id = a.id
        AND umls.id = t.cid
        GROUP BY TO_CHAR(app.priority_date, 'YYYY'), umls.canonical_name
        ORDER BY weight DESC LIMIT 1000
    """

    relationships = PsqlDatabaseClient().select(sql)
    g = generate_graph(relationships, max_nodes=max_nodes)

    # create serialized link data
    link_data = nx.node_link_data(g)

    return SerializableGraph(
        edges=[
            Link(source=l["source"], target=l["target"], weight=l["weight"])
            for l in sorted(link_data["links"], key=lambda x: x["weight"], reverse=True)
        ],
        nodes=[
            Node(
                id="root",
                label="root",
                parent="",
                size=1,
                group="",
            ),
            Node(
                id="entity",
                label="entity",
                parent="root",
                size=1,
                group="root",
            ),
            Node(
                id="patent",
                label="patent",
                parent="root",
                size=1,
                group="root",
            ),
            *sorted(
                [
                    Node(
                        id=n["id"],
                        label=n["id"],
                        parent=n["group"],
                        size=n["size"],
                        group=n["group"],
                    )
                    for n in link_data["nodes"]
                ],
                key=lambda x: x.size,
                reverse=True,
            ),
        ],
    )
