"""
Functions to initialize the patents database
"""
import logging

from system import initialize

initialize()

from clients.low_level.big_query import (
    delete_bg_table,
    query_to_bg_table,
    BQ_DATASET_ID,
)

from .copy_tables import copy_patent_tables
from .terms import create_patent_terms

logging.basicConfig(level=logging.INFO)

FIELDS = [
    # gpr_publications
    "gpr_pubs.publication_number as publication_number",
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
]


def __create_applications_table():
    """
    Create a table of patent applications for use in app queries
    """
    logging.info("Create a table of patent applications for use in app queries")
    applications = f"""
        SELECT
        {','.join(FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` as pubs,
        `{BQ_DATASET_ID}.gpr_publications` as gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    query_to_bg_table(applications, "applications")


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logging.info("Create a table of annotations for use in app queries")
    table_id = "annotations"
    delete_bg_table(table_id)

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
                    ROW_NUMBER() OVER(PARTITION BY publication_number, LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) ORDER BY character_offset_start) as rank
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
    query_to_bg_table(entity_query, table_id)


def main():
    """
    Copy tables from patents-public-data to a local dataset.
    Order matters. Non-idempotent.

    Usage:
        >>> python3 -m scripts.patents.initialize_patents
    """
    # copy gpr_publications, publications, gpr_annotations tables
    # copy_patent_tables()

    # create terms and synonym map tables
    # create_patent_terms()

    # create the (small) tables against which the app will query
    __create_applications_table()
    __create_annotations_table()


if __name__ == "__main__":
    main()
