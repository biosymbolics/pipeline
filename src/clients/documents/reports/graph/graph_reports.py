"""
Patent graph reports
"""

from enum import Enum
from typing import Sequence
import logging
import networkx as nx
from pydash import uniq
import polars as pl
from prisma.enums import BiomedicalEntityType

from clients.low_level.prisma import prisma_client
from typings import (
    DOMAINS_OF_INTEREST,
    DocType,
    ScoredTrial,
    ScoredRegulatoryApproval,
    ScoredPatent,
)

from .types import (
    AggregateDocumentRelationship,
    Node,
    Link,
    SerializableGraph,
)

from ..constants import X_DIMENSIONS, Y_DIMENSIONS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NODES = 100
MIN_NODE_DEGREE = 2
MAX_TAILS = 50
MAX_HEADS = 20

RELATIONSHIPS_OF_INTEREST = [
    "allelic_variant_of",
    "biological_process_involves_gene_product",
    "chemical_or_drug_has_mechanism_of_action",
    "chemical_or_drug_has_physiologic_effect",
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
    # "has_phenotype",
    # "has_manifestation",
    "has_gene_product_element",
    "has_therapeutic_class",
    "has_mechanism_of_action",
    "has_physiologic_effect",
    "has_target",
    # "is_mechanism_of_action_of_chemical_or_drug",
    # "is_physiologic_effect_of_chemical_or_drug",
    "is_target",
    # "is_target_of",
    # "may_treat",
    # "may_be_treated_by",
    # "manifestation_of",
    # "mechanism_of_action_of",
    # "molecular_abnormality_involves_gene",
    "negatively_regulates",
    "negatively_regulated_by",
    "pathogenesis_of_disease_involves_gene",
    "pathway_has_gene_element",
    # "phenotype_of",
    # "physiologic_effect_of",
    "positively_regulates",
    "positively_regulated_by",
    "process_involves_gene",
    "related_to_genetic_biomarker",
    "regulated_by",
    "regulates",
    "therapeutic_class_of",
]

ENTITY_GROUP = "entity"
DOCUMENT_GROUP = "patent"


class TermField(Enum):
    canonical_name = "canonical_name"
    instance_rollup = "instance_rollup"


def get_entity_subquery(term_field: TermField, doc_type: DocType) -> str:
    """
    Subquery to use if the term_field is an entity (indicatable, intervenable)

    Args:
        term_field (TermField): term field
        doc_type (DocType): doc type
    """
    return f"""
        (
            select {doc_type.name}_id, entity_id, canonical_type, {term_field.name}
            from intervenable
            where {doc_type.name}_id is not null

            UNION ALL

            select {doc_type.name}_id, entity_id, canonical_type, {term_field.name}
            from indicatable
            where {doc_type.name}_id is not null
        )
    """


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
                *[n[0] for n in list(g.nodes(data="group")) if n[1] == DOCUMENT_GROUP],  # type: ignore
                *top_nodes,
            ]
        )
    )

    return subgraph


async def graph_document_relationships(
    ids: Sequence[str],
    doc_type: DocType = DocType.patent,
    head_field: str = "priority_date",
    max_nodes: int = MAX_NODES,
) -> SerializableGraph:
    """
    Graph UMLS ancestory for a set of documents

    - Nodes are UMLS entities
    - Edges are ancestory relationships OR patent co-occurrences (e.g. PD and anti-alpha-synucleins)
    - Nodes contain metadata of the relevant documents

    Args:
        ids (Sequence[str]): doc ids (must correspond to doc_type)
        doc_type (DocType, optional): doc type. Defaults to DocType.patent.
        head_field (str, optional): head field. Defaults to "priority_date".
        max_nodes (int, optional): max nodes. Defaults to MAX_NODES.
    """
    head_field_info = Y_DIMENSIONS[doc_type][head_field]
    head_sql = head_field_info.transform(head_field)

    # TODO: save relationship type, ancestors
    sql = f"""
        -- documents-node to entity relationships
        SELECT
            {head_sql} as head,
            umls.name as tail,
            count(*) as weight,
            '{DOCUMENT_GROUP}' as group
        FROM {doc_type.name}
            JOIN {get_entity_subquery(TermField.canonical_name, doc_type)} entities
                ON entities.{doc_type.name}_id = {doc_type.name}.id
            JOIN biomedical_entity ON
                biomedical_entity.id = entities.canonical_id
                AND biomedical_entity.entity_types in {tuple(DOMAINS_OF_INTEREST)}
            JOIN _entity_to_umls as etu ON etu."A"=biomedical_entity.id
            JOIN umls ON umls.id = etu."B"
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        GROUP BY {head_sql}, umls.name
        ORDER BY weight DESC LIMIT 1000
    """

    client = await prisma_client(300)
    relationships = await client.query_raw(sql)
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


async def aggregate_document_relationships(
    ids: Sequence[str],
    head_field: str = "priority_date",
    entity_types: Sequence[BiomedicalEntityType] | None = None,
    relationships: Sequence[str] = RELATIONSHIPS_OF_INTEREST,
    doc_type: DocType = DocType.patent,
) -> list[AggregateDocumentRelationship]:
    """
    Aggregated UMLS ancestory report for a set of documents

    Args:
        ids (Sequence[str]): doc ids (must correspond to doc_type)
        head_field (str, optional): head field. Defaults to "priority_date".
        entity_types (Sequence[BiomedicalEntityType], optional): entity types. Defaults to None.
        relationships (Sequence[str], optional): relationships. Defaults to RELATIONSHIPS_OF_INTEREST.
        doc_type (DocType, optional): doc type. Defaults to DocType.patent.
    """

    if head_field not in Y_DIMENSIONS[doc_type]:
        raise ValueError(f"Invalid head field: {head_field}")

    head_field_info = Y_DIMENSIONS[doc_type][head_field]
    head_sql = head_field_info.transform(head_field)

    entity_sq = get_entity_subquery(TermField.canonical_name, doc_type)
    sql = f"""
        -- document to entity relationships
        SELECT
            {head_sql} as head,
            umls.name as concept,
            count(*) as count,
            array_agg(distinct {doc_type.name}.id) as documents
        FROM {doc_type.name}
            JOIN {entity_sq} entities
                ON entities.{doc_type.name}_id = {doc_type.name}.id
            JOIN biomedical_entity ON biomedical_entity.id = entities.entity_id
                {f"AND biomedical_entity.entity_types in {tuple(entity_types)} " if entity_types else ""}
            JOIN _entity_to_umls as etu ON etu."A"=biomedical_entity.id
            JOIN umls ON umls.id = etu."B"
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        GROUP BY {head_sql}, concept

        UNION ALL

        -- entity to entity relationships
        SELECT
            {head_sql} as head,
            tail_name as concept,
            count(*) as count,
            array_agg(distinct {doc_type.name}.id) as documents
        FROM {doc_type.name}
            JOIN {entity_sq} entities ON entities.{doc_type.name}_id = {doc_type.name}.id
            JOIN biomedical_entity ON biomedical_entity.id = entities.entity_id
                {f"AND biomedical_entity.entity_types in {tuple(entity_types)} " if entity_types else ""}
            JOIN _entity_to_umls as etu ON etu."A"=biomedical_entity.id
            JOIN umls_graph ON umls_graph.head_id = etu."B"
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        AND head_id<>tail_id
        AND umls_graph.relationship in {tuple(relationships)}
        GROUP BY {head_sql}, concept
    """

    client = await prisma_client(300)
    results = await client.query_raw(sql)
    df = pl.DataFrame(results)

    # get top concepts (i.e. UMLS terms represented across as many of the head dimension as possible)
    top_concepts = (
        df.group_by("concept")
        .agg(pl.count("head").alias("head_count"))
        .sort(pl.col("head_count"), descending=True)
        .limit(MAX_TAILS)
    )
    top_heads = (
        df.group_by("head")
        .agg(pl.count("concept").alias("concept_count"))
        .sort(pl.col("concept_count"), descending=True)
        .limit(MAX_HEADS)
    )
    top_records = (
        df.join(top_concepts, on="concept", how="inner")
        .join(top_heads, on="head", how="inner")
        .drop(["head_count", "concept_count"])
        .to_dicts()
    )

    return [AggregateDocumentRelationship(**tr) for tr in top_records]
