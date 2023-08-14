"""
Functions for copying around subsets of the patents database
"""
from clients.low_level.big_query import (
    DatabaseClient,
    BQ_DATASET_ID,
)
from scripts.patents.copy_psql import copy_patent_approvals

from .gpr_constants import COMMON_ENTITY_NAMES, SUPPRESSED_DOMAINS

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))


def __copy_gpr_publications():
    """
    Copy publications from GPR to a local table
    """
    table_id = "gpr_publications"
    client = DatabaseClient()
    client.delete_table(table_id)

    query = f"""
        SELECT * FROM `patents-public-data.google_patents_research.publications`
        WHERE EXISTS
        (SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))
    """
    client.query_to_table(query, table_id)


def __copy_gpr_annotations():
    """
    Copy annotations from GPR to a local table

    To remove annotations after load:
    ``` sql
    UPDATE `patents.entities`
    SET annotations = ARRAY(
        SELECT AS STRUCT *
        FROM UNNEST(annotations) as annotation
        WHERE annotation.domain NOT IN ('chemClass', 'chemGroup', 'anatomy')
    )
    WHERE EXISTS(
        SELECT 1
        FROM UNNEST(annotations) AS annotation
        WHERE annotation.domain IN ('chemClass', 'chemGroup', 'anatomy')
    )
    ```

    or from gpr_annotations:
    ``` sql
    DELETE FROM `fair-abbey-386416.patents.gpr_annotations` where domain in
    ('chemClass', 'chemGroup', 'anatomy') OR preferred_name in ("seasonal", "behavioural", "mental health")
    ```
    """
    table_id = "gpr_annotations"
    client = DatabaseClient()
    client.delete_table(table_id)

    query = f"""
        SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations
        JOIN `{BQ_DATASET_ID}.publications` AS local_publications
        ON local_publications.publication_number = annotations.publication_number
        WHERE annotations.confidence > 0.69
        AND LOWER(preferred_name) not in {COMMON_ENTITY_NAMES}
        AND domain not in {SUPPRESSED_DOMAINS}
    """
    client.query_to_table(query, table_id)


def __copy_publications():
    """
    Copy publications from patents-public-data to a local table

    NOTE: this has not been tested
    """
    table_id = "publications"
    client = DatabaseClient()
    client.delete_table(table_id)

    # add to this all publication_numbers with the same family_id
    query = f"""
        WITH numbered_rows AS (
        SELECT *,
        ROW_NUMBER() OVER (PARTITION BY publication_number) as row_number
        FROM (
            SELECT
            main_publications.*,
            ARRAY_AGG(related_publications.publication_number) OVER (PARTITION BY main_publications.family_id) AS all_publication_numbers
            FROM `patents-public-data.patents.publications` as main_publications
            JOIN `{BQ_DATASET_ID}.gpr_publications` AS local_gpr
            ON local_gpr.publication_number = main_publications.publication_number
            JOIN `patents-public-data.patents.publications` AS related_publications
            ON related_publications.family_id = main_publications.family_id
            WHERE main_publications.application_kind = 'W'
        )
        )
        SELECT *
        FROM numbered_rows
        WHERE row_number = 1
    """
    client.query_to_table(query, table_id)


def copy_patent_tables():
    """
    Copy tables from patents-public-data to a local dataset

    Order matters.
    Idempotent (as in, tables are deleted and recreated) but not atomic, and also expensive (from a BigQuery standpoint)
    """
    # copy gpr_publications table
    __copy_gpr_publications()

    # copy publications table
    __copy_publications()

    # copy gpr_annotations table
    __copy_gpr_annotations()

    # copy data about approvals
    copy_patent_approvals()
