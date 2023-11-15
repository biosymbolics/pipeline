"""
Patent graph reports
"""

from typing import Sequence
import logging
import networkx as nx

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from clients.patents.constants import ENTITY_DOMAINS
from constants.core import ANNOTATIONS_TABLE, TERM_IDS_TABLE
from typings.patents import PatentApplication

from .types import Node, Link, SerializableGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NODES = 250

RELATIONSHIPS_OF_INTEREST = [
    "allelic_variant_of",
    "biological_process_involves_gene_product",
    "chemical_or_drug_has_mechanism_of_action",
    "chemical_or_drug_has_physiologic_effect",
    "genetic_biomarker_related_to",
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
    "has_manifestation",
    "has_gene_product_element",
    "has_therapeutic_class",
    "has_mechanism_of_action",
    "has_physiologic_effect",
    "has_target",
    "is_mechanism_of_action_of_chemical_or_drug",
    "is_physiologic_effect_of_chemical_or_drug",
    "is_target",
    "is_target_of",
    "may_treat",
    "may_be_treated_by",
    "manifestation_of",
    "mechanism_of_action_of",
    "molecular_abnormality_involves_gene",
    "negatively_regulates",
    "negatively_regulated_by",
    "pathogenesis_of_disease_involves_gene",
    "pathway_has_gene_element",
    "phenotype_of",
    "physiologic_effect_of",
    "positively_regulates",
    "positively_regulated_by",
    "process_involves_gene",
    "related_to_genetic_biomarker",
    "regulated_by",
    "regulates",
    "therapeutic_class_of",
]


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
    node_to_patents = {r["head"]: r["patent_ids"] for r in relationships}
    nx.set_node_attributes(g, node_to_patents, "patent_ids")

    degrees: list[tuple[str, number]] = g.degree  # type: ignore

    # take the top `max_node` nodes by degree; create subgraph
    top_nodes = [
        node
        for (node, _) in sorted(degrees, key=lambda x: x[1], reverse=True)[:max_nodes]
    ]
    return g.subgraph(top_nodes)


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
        -- direct relationships
        SELECT
            head_name as head,
            tail_name as tail,
            count(*) as weight,
            array_agg(publication_number) as patent_ids
        FROM {ANNOTATIONS_TABLE} a, {TERM_IDS_TABLE} t, umls_graph g
        WHERE a.publication_number = ANY(ARRAY{patent_ids})
        AND a.domain in {tuple(ENTITY_DOMAINS)}
        AND t.id = a.id
        AND g.head_id = t.cid
        AND head_id<>tail_id
        AND g.relationship in {tuple(RELATIONSHIPS_OF_INTEREST)}
        GROUP BY head_name, tail_name, g.relationship
    """
    # """
    # -- ancestor relationships (but only category_rollup for now)
    # SELECT
    #     head_umls.canonical_name as head,
    #     tail_umls.canonical_name as tail,
    #     count(*) as weight,
    #     array_agg(publication_number) as patent_ids
    # FROM
    # {ANNOTATIONS_TABLE} a,
    # {TERM_IDS_TABLE} t,
    # umls_lookup head_umls,
    # umls_lookup tail_umls
    # WHERE a.publication_number = ANY(ARRAY{patent_ids})
    # AND a.domain in {tuple(ENTITY_DOMAINS)}
    # AND t.id = a.id
    # AND head_umls.id = t.cid
    # AND head_umls.category_rollup <> t.cid -- no self-loops
    # AND tail_umls.id = head_umls.category_rollup
    # GROUP BY head, tail
    #
    #     UNION ALL
    #     -- co-occurrence relationships
    #     SELECT
    #         a1.term AS head,
    #         a2.term AS tail,
    #         count(*) AS weight,
    #         array_agg(a1.publication_number) AS patent_ids
    #     FROM {ANNOTATIONS_TABLE} a1
    #     JOIN {ANNOTATIONS_TABLE} a2 ON a1.publication_number = a2.publication_number
    #     WHERE a1.publication_number = ANY(ARRAY{patent_ids})
    #     AND a1.domain in {tuple(ENTITY_DOMAINS)}
    #     AND a2.domain in {tuple(ENTITY_DOMAINS)}
    #     AND a1.term < a2.term
    #     GROUP BY a1.term, a2.term
    # """
    relationships = PsqlDatabaseClient().select(sql)
    g = generate_graph(relationships, max_nodes=max_nodes)

    # create serialized link data
    link_data = nx.node_link_data(g)

    return SerializableGraph(
        edges=[
            Link(source=l["source"], target=l["target"], weight=2)
            for l in link_data["links"]
        ],
        nodes=[
            Node(
                id=n["id"],
                label=n["id"],
                patent_ids=n.get("patent_ids", []),
                size=len(n.get("patent_ids", [])),
            )
            for n in link_data["nodes"]
        ],
    )
