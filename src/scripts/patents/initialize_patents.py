"""
Functions to initialize the patents database
"""
import logging
import sys
from google.cloud import bigquery

from system import initialize

initialize()

from constants.patents import PATENT_ATTRIBUTE_MAP
from clients.low_level.big_query import (
    create_bq_table,
    delete_bg_table,
    query_to_bg_table,
    BQ_DATASET_ID,
)
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)

from .copy_tables import copy_patent_tables
from .terms import create_patent_terms

logging.basicConfig(level=logging.INFO)


FIELDS = [
    # gpr_publications
    "gpr_pubs.publication_number as publication_number",
    "regexp_replace(gpr_pubs.publication_number, '-[^-]*$', '') as base_publication_number",  # for matching against approvals
    "abstract",
    "application_number",
    "cited_by",
    "country",
    "embedding_v1 as embeddings",
    "similar",
    "title",
    "top_terms",
    "url",
    # publications
    "application_kind",
    "assignee as assignee_raw",
    "assignee_harmonized",
    "ARRAY(SELECT assignee.name FROM UNNEST(assignee_harmonized) as assignee) as assignees",
    "citation",
    "claims_localized as claims",
    "ARRAY(SELECT cpc.code FROM UNNEST(pubs.cpc) as cpc) as cpc_codes",
    "family_id",
    "filing_date",
    "grant_date",
    "inventor as inventor_raw",
    "inventor_harmonized",
    "ARRAY(SELECT inventor.name FROM UNNEST(inventor_harmonized) as inventor) as inventors",
    "ARRAY(SELECT ipc.code FROM UNNEST(ipc) as ipc) as ipc_codes",
    "kind_code",
    "pct_number",
    "priority_claim",
    "priority_date",
    "publication_date",
    "spif_application_number",
    "spif_publication_number",
    "all_publication_numbers",
    "ARRAY(select regexp_replace(pn, '-[^-]*$', '') from UNNEST(all_publication_numbers) as pn) as all_base_publication_numbers",
]


def __create_applications_table():
    """
    Create a table of patent applications for use in app queries
    """
    logging.info("Create a table of patent applications for use in app queries")

    table_name = "applications"
    delete_bg_table(table_name)

    applications = f"""
        SELECT
        {','.join(FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` as pubs,
        `{BQ_DATASET_ID}.gpr_publications` as gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    query_to_bg_table(applications, table_name)


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logging.info("Create a table of annotations for use in app queries")
    table_name = "annotations"
    delete_bg_table(table_name)

    entity_query = f"""
        WITH ranked_terms AS (
                --- existing annotations
                SELECT
                    publication_number,
                    ocid,
                    LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) as term,
                    domain,
                    confidence,
                    source,
                    character_offset_start,
                    ROW_NUMBER() OVER(
                        PARTITION BY publication_number, LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name))
                        ORDER BY character_offset_start
                    ) as rank
                FROM `{BQ_DATASET_ID}.gpr_annotations` a
                LEFT JOIN `{BQ_DATASET_ID}.synonym_map` map ON LOWER(a.preferred_name) = map.synonym

                UNION ALL

                --- assignees as annotations
                SELECT
                    publication_number,
                    0 as ocid,
                    LOWER(IF(map.term IS NOT NULL, map.term, assignee.name)) as term,
                    "assignee" as domain,
                    1.0 as confidence,
                    "record" as source,
                    1 as character_offset_start,
                    1 as rank
                FROM `{BQ_DATASET_ID}.publications` p,
                unnest(p.assignee_harmonized) as assignee
                LEFT JOIN `{BQ_DATASET_ID}.synonym_map` map ON LOWER(assignee.name) = map.synonym

                UNION ALL

                --- inventors as annotations
                SELECT
                    publication_number,
                    0 as ocid,
                    LOWER(IF(map.term IS NOT NULL, map.term, inventor.name)) as term,
                    "inventor" as domain,
                    1.0 as confidence,
                    "record" as source,
                    1 as character_offset_start,
                    1 as rank
                FROM `{BQ_DATASET_ID}.publications` p,
                unnest(p.inventor_harmonized) as inventor
                LEFT JOIN `{BQ_DATASET_ID}.synonym_map` map ON LOWER(inventor.name) = map.synonym

                UNION ALL

                --- biosym (our) annotations
                SELECT
                    publication_number,
                    0 as ocid,
                    LOWER(COALESCE(NULLIF(syn_map.term, ''), NULLIF(a.canonical_term, ''), a.original_term)) as term,
                    domain,
                    confidence,
                    source,
                    character_offset_start,
                    1 as rank
                FROM `{WORKING_BIOSYM_ANNOTATIONS_TABLE}` a
                LEFT JOIN `{BQ_DATASET_ID}.synonym_map` syn_map ON LOWER(COALESCE(NULLIF(a.canonical_term, ''), a.original_term)) = syn_map.synonym
        )
        SELECT
            publication_number,
            ARRAY_AGG(
                struct(ocid, term, domain, confidence, source, character_offset_start)
                ORDER BY character_offset_start
            ) as annotations
        FROM ranked_terms
        WHERE rank = 1
        GROUP BY publication_number
    """
    query_to_bg_table(entity_query, table_name)


def __create_biosym_annotations_tables():
    """
    Creates biosym annotations tables if need be
    (NOTE: does not check schema if exists)
    """
    table_names = [SOURCE_BIOSYM_ANNOTATIONS_TABLE, WORKING_BIOSYM_ANNOTATIONS_TABLE]

    for table_name in table_names:
        schema = [
            bigquery.SchemaField("publication_number", "STRING"),
            bigquery.SchemaField("original_term", "STRING"),
            bigquery.SchemaField("domain", "STRING"),
            bigquery.SchemaField("confidence", "FLOAT"),
            bigquery.SchemaField("source", "STRING"),
            bigquery.SchemaField("character_offset_start", "INTEGER"),
            bigquery.SchemaField("character_offset_end", "INTEGER"),
        ]
        create_bq_table(table_name, schema, exists_ok=True, truncate_if_exists=False)
        logging.info(f"(Maybe) created table {table_name}")


def main(copy_tables: bool = False):
    """
    Copy tables from patents-public-data to a local dataset.
    Order matters.

    Usage:
        >>> python3 -m scripts.patents.initialize_patents -copy_tables
        >>> python3 -m scripts.patents.initialize_patents
    """
    # __create_biosym_annotations_tables()

    if copy_tables:
        # copy gpr_publications, publications, gpr_annotations tables
        # idempotent but expensive
        copy_patent_tables()

    # create small-ish table of patent applications
    # __create_applications_table()

    # create patent terms
    create_patent_terms()

    # create annotations
    __create_annotations_table()

    # TODODODOD!!
    # patent_attributes = classify_by_keywords(titles, PATENT_ATTRIBUTE_MAP, None)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.initialize_patents [-copy_tables]\nCopies and transforms patent data"
        )
        sys.exit()

    copy_tables: bool = "copy_tables" in sys.argv
    main(copy_tables)
