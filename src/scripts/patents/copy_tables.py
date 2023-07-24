"""
Functions for copying around subsets of the patents database
"""
from clients.low_level.big_query import (
    delete_bg_table,
    query_to_bg_table,
    BQ_DATASET_ID,
)
from scripts.patents.copy_psql import copy_patent_approvals

from ._constants import COMMON_ENTITY_NAMES, SUPPRESSED_DOMAINS

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))


def __copy_gpr_publications():
    """
    Copy publications from GPR to a local table
    """
    table_id = "gpr_publications"
    delete_bg_table(table_id)

    query = f"""
        SELECT * FROM `patents-public-data.google_patents_research.publications`
        WHERE EXISTS
        (SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))
    """
    query_to_bg_table(query, table_id)


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
    delete_bg_table(table_id)

    query = f"""
        SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations
        JOIN `{BQ_DATASET_ID}.publications` AS local_publications
        ON local_publications.publication_number = annotations.publication_number
        WHERE annotations.confidence > 0.69
        AND LOWER(preferred_name) not in {COMMON_ENTITY_NAMES}
        AND domain not in {SUPPRESSED_DOMAINS}
    """
    query_to_bg_table(query, table_id)


def __copy_publications():
    """
    Copy publications from patents-public-data to a local table
    """
    table_id = "publications"
    delete_bg_table(table_id)

    # add to this all publication_numbers with the same family_id
    query = f"""
        SELECT
            main_publications.*,
            ARRAY_AGG(related_publications.publication_number) OVER (PARTITION BY main_publications.family_id) AS all_publication_numbers
        FROM `patents-public-data.patents.publications` as main_publications
        JOIN `{BQ_DATASET_ID}.gpr_publications` AS local_gpr
        ON local_gpr.publication_number = main_publications.publication_number
        JOIN `patents-public-data.patents.publications` AS related_publications
        ON related_publications.family_id = main_publications.family_id
        WHERE main_publications.application_kind = 'W'
    """
    query_to_bg_table(query, table_id)


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

    copy_patent_approvals()
