"""
Patent graph reports
"""

import time
from typing import Sequence
import logging
import polars as pl
from prisma.enums import BiomedicalEntityType

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from constants.core import SEARCH_TABLE
from constants.umls import CATEGORY_TO_ENTITY_TYPES
from typings import DocType
from typings.client import DocumentCharacteristicParams
from utils.string import get_id

from .types import AggregateDocumentRelationship

from ..constants import Y_DIMENSIONS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_NODES = 100
MIN_NODE_DEGREE = 2
MAX_TAILS = 30
MAX_HEADS = 15


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
        "has_structural_class",
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
        "has_active_moiety",  # glucoses -> glucose 50 MG/ML Injection
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


def _apply_limit(df: pl.DataFrame) -> list[AggregateDocumentRelationship]:
    top_tails = (
        df.group_by("tail")
        .agg(pl.count("head").alias("head_count"))
        .sort(pl.col("head_count"), descending=True)
        .limit(MAX_TAILS)
    )
    top_heads = (
        df.group_by("head")
        .agg(pl.count("tail").alias("tail_count"))
        .sort(pl.col("tail_count"), descending=True)
        .limit(MAX_HEADS)
    )
    top_records = (
        df.join(top_tails, on="tail", how="inner")
        .join(top_heads, on="head", how="inner")
        .drop(["head_count", "tail_count"])
        .to_dicts()
    )
    return [AggregateDocumentRelationship(**tr) for tr in top_records]


async def _aggregate_document_umls_relationships(
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

    sql = f"""
        -- document to entity relationships
        SELECT
            {head_sql} as head,
            umls.name as tail,
            count(*) as count,
            array_agg(distinct {doc_type.name}.id) as documents
        FROM {doc_type.name}
            JOIN {SEARCH_TABLE} ON {SEARCH_TABLE}.{doc_type.name}_id = {doc_type.name}.id
                {f"AND types in {tuple(entity_types)} " if entity_types else ""}
            JOIN _entity_to_umls as etu ON etu."A"={SEARCH_TABLE}.entity_id
            JOIN umls ON umls.id = etu."B"
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        GROUP BY {head_sql}, tail

        UNION ALL

        -- entity to entity relationships
        SELECT
            {head_sql} as head,
            tail_name as tail,
            count(*) as count,
            array_agg(distinct {doc_type.name}.id) as documents
        FROM {doc_type.name}
            JOIN {SEARCH_TABLE} ON {SEARCH_TABLE}.{doc_type.name}_id = {doc_type.name}.id
                {f"AND types in {tuple(entity_types)} " if entity_types else ""}
            JOIN _entity_to_umls as etu ON etu."A"={SEARCH_TABLE}.entity_id
            JOIN umls_graph ON umls_graph.tail_id = etu."B" -- note matching umls *tail_id* to entity, thus looking for heads.
        WHERE {doc_type.name}.id = ANY(ARRAY{ids})
        AND head_id<>tail_id
        AND umls_graph.relationship in {tuple(relationships)}
        GROUP BY {head_sql}, tail
    """

    async with prisma_context(300) as db:
        results = await db.query_raw(sql)

    # get top tail (i.e. UMLS terms represented across as many of the head dimension as possible)
    top_records = _apply_limit(pl.DataFrame(results))

    logger.info(
        "Generated characteristics report (%s records) in %s seconds",
        len(results),
        round(time.monotonic() - start),
    )

    return top_records


async def _aggregate_document_entity_relationships(
    p: DocumentCharacteristicParams,
) -> list[AggregateDocumentRelationship]:
    """
    Aggregated intervention x indication report for a set of documents
    """
    sql = f"""
        SELECT
            head.category_rollup head,
            tail.category_rollup tail,
            count(distinct head.{p.doc_type.name}_id) AS count,
            array_agg(distinct head.{p.doc_type.name}_id) AS documents
        FROM
            {SEARCH_TABLE} search_table,
            {SEARCH_TABLE} head,
            {SEARCH_TABLE} tail
        WHERE search_table.search @@ plainto_tsquery('english', $1)
        AND search_table.{p.doc_type.name}_id=head.{p.doc_type.name}_id
        AND head.{p.doc_type.name}_id = tail.{p.doc_type.name}_id
        AND head.type = ANY($2::"BiomedicalEntityType"[])
        AND tail.type = ANY($3::"BiomedicalEntityType"[])
        GROUP BY head.category_rollup, tail.category_rollup
        ORDER BY count(*) DESC
    """
    start = time.monotonic()

    # TODO - AND/OR; description
    term_query = " ".join(p.terms)

    async with prisma_context(300) as db:
        results = await db.query_raw(
            sql,
            term_query,
            CATEGORY_TO_ENTITY_TYPES[p.head_field],
            CATEGORY_TO_ENTITY_TYPES[p.tail_field],
        )

    top_records = _apply_limit(pl.DataFrame(results))

    logger.info(
        "Generated entity relationships report (%s records) in %s seconds",
        len(top_records),
        round(time.monotonic() - start),
    )

    return top_records


async def aggregate_document_relationships(
    p: DocumentCharacteristicParams,
) -> list[AggregateDocumentRelationship]:
    """
    Aggregated UMLS ancestory report for a set of documents
    """

    async def _run_report() -> list[AggregateDocumentRelationship]:
        return await _aggregate_document_entity_relationships(p=p)

    if p.skip_cache == True:
        patents = await _run_report()
        return patents

    key = get_id({**p.model_dump(), "api": "document_relationships"})

    return await retrieve_with_cache_check(
        _run_report,
        key=key,
        decode=lambda str_data: storage_decoder(str_data),
    )
