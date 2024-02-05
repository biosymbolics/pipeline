"""
Patent graph reports
"""

import time
from typing import Sequence
import logging
import networkx as nx
from pydash import uniq
import polars as pl
from prisma.enums import BiomedicalEntityType

from clients.documents.constants import DOC_CLIENT_LOOKUP
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_client
from typings import (
    DOMAINS_OF_INTEREST,
    DocType,
    TermField,
)
from typings.client import DocumentCharacteristicParams
from utils.string import get_id

from .types import (
    AggregateDocumentRelationship,
    Node,
    Link,
    SerializableGraph,
)

from ..constants import Y_DIMENSIONS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NODES = 100
MIN_NODE_DEGREE = 2
MAX_TAILS = 50
MAX_HEADS = 20


# IMPORTANT: these only make sense if "head" is a *parent* and not a child of the document.
# I.E. less specific, not more specific.
RELATIONSHIPS_OF_INTEREST_MAP = {
    "DISEASE": [
        "has_phenotype",  # head/disease->tail/phenotype, e.g. Mantle-Cell Lymphoma -> (specific MCL phenotype)
        "has_manifestation",  # Pains, Abdominal -> fabrys disease
        "may_treat",  # Pains, Abdominal -> ALOSETRON HYDROCHLORIDE
        "gene_involved_in_molecular_abnormality",  # Philadelphia Chromosome -> BCR/ABL FUSION GENE
        "gene_involved_in_pathogenesis_of_disease",
    ],
    "MECHANISM": [
        "has_physiologic_effect",  # head/effect -> tail/drug, e.g. Increased Glycolysis [PE]  -> PIOGLITAZONE
        "has_mechanism_of_action",  # head/MoA->tail/drug
        "has_therapeutic_class",  # e.g. Antihelmintics -> ALBENDAZOLE
        "negatively_regulates",  # neurotransmitter uptake -> negative regulation of neurotransmitter uptake
        "positively_regulates",  # neurotransmitter uptake -> positive regulation of neurotransmitter uptake
        "chemical_or_drug_has_mechanism_of_action",  # Regulation, Gene Expression -> RESVERATROL
        "chemical_or_drug_has_physiologic_effect",
        "regulates",
        "gene_plays_role_in_process",  # Acetylations -> NAT2 Gene
        "gene_product_plays_role_in_process",
    ],
    "BIOLOGIC": [
        "has_target",
        "allelic_variant_of",  # REL Gene -> REL, IVS5DS, G-A, +1
        "pathogenesis_of_disease_involves_gene",  # TP53 Genes -> Colorectal Carcinomas
        "molecular_abnormality_involves_gene",  # TP53 Genes -> TP53 Deleterious Gene Mutation
        "pathogenesis_of_disease_involves_gene",
        "process_involves_gene",
        "related_to_genetic_biomarker",  # BCL2 Gene -> BCL2 Overexpression Positive
        "gene_product_encoded_by_gene",  # MHC Class II Genes -> HLA-DP Antigens
        "gene_product_variant_of_gene_product",  # TP53 protein, human -> TP53 NP_000537.3:p.Y220C
    ],
    # this goes from non-specific to very specific (thus not included with rest of RELATIONSHIPS_OF_INTEREST)
    "COMPOUND": [
        "may_be_treated_by",  # ALOSETRON HYDROCHLORIDE -> Pains, Abdominal
        "mechanism_of_action_of",  # PIOGLITAZONE -> Increased Glycolysis
        "therapeutic_class_of",  # ALBENDAZOLE -> Antihelmintics
        "has_active_ingredient",  # DROXIDOPA -> DROXIDOPA 100 mg ORAL CAPSULE
    ],
}

RELATIONSHIPS_OF_INTEREST = [
    "isa",
    *RELATIONSHIPS_OF_INTEREST_MAP["DISEASE"],
    *RELATIONSHIPS_OF_INTEREST_MAP["MECHANISM"],
    *RELATIONSHIPS_OF_INTEREST_MAP["BIOLOGIC"],
]

ENTITY_GROUP = "entity"
DOCUMENT_GROUP = "patent"


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

    client = await prisma_client(120)
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


async def _aggregate_document_relationships(
    ids: Sequence[str],
    head_field: str,
    entity_types: Sequence[BiomedicalEntityType] | None,
    relationships: Sequence[str],
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
    start = time.monotonic()

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
            JOIN umls_graph ON umls_graph.tail_id = etu."B" -- note matching umls *tail_id* to entity, thus looking for heads.
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        AND head_id<>tail_id
        AND umls_graph.relationship in {tuple(relationships)}
        GROUP BY {head_sql}, concept
    """

    client = await prisma_client(120)
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

    logger.info(
        "Generated characteristics report (%s records) in %s seconds",
        len(results),
        round(time.monotonic() - start),
    )

    return [AggregateDocumentRelationship(**tr) for tr in top_records]


async def aggregate_document_relationships(
    p: DocumentCharacteristicParams,
) -> list[AggregateDocumentRelationship]:
    """
    Aggregated UMLS ancestory report for a set of documents
    """

    async def _run_report() -> list[AggregateDocumentRelationship]:
        documents = await DOC_CLIENT_LOOKUP[p.doc_type].search(p)
        if len(documents) == 0:
            logging.info("No documents found for terms: %s", p.terms)
            return []

        return await _aggregate_document_relationships(
            ids=[d.id for d in documents],
            head_field=p.head_field,
            entity_types=None,
            relationships=RELATIONSHIPS_OF_INTEREST,
            doc_type=p.doc_type,
        )

    if p.skip_cache == True:
        patents = await _run_report()
        return patents

    key = get_id({**p.__dict__, "api": "document_relationships"})

    return await retrieve_with_cache_check(
        _run_report,
        key=key,
        decode=lambda str_data: storage_decoder(str_data),
    )
