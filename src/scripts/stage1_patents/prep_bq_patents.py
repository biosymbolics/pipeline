"""
Functions for copying around subsets of the patents database
"""

from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID
from constants.core import GPR_ANNOTATIONS_TABLE
from constants.patents import BIOMEDICAL_IPC_CODE_PREFIX_RE
from .constants import GPR_PUBLICATIONS_TABLE


async def _copy_publications():
    """
    Copy publications from `patents-public-data` to a BigQuery table in our workspace
    """
    table_id = "publications"
    client = BQDatabaseClient()
    await client.delete_table(table_id)

    # adds all publication_numbers with the same family_id
    query = f"""
        WITH numbered_rows AS (
            SELECT *,
            ROW_NUMBER() OVER (PARTITION BY publication_number) as row_number
            FROM (
                SELECT main_publications.*,
                ARRAY_CONCAT(related_publications.claims_localized) as claim_texts,
                ARRAY_AGG(related_publications.publication_number)
                    OVER (PARTITION BY main_publications.family_id) AS all_publication_numbers
                FROM `patents-public-data.patents.publications` AS main_publications
                JOIN `patents-public-data.patents.publications` AS related_publications
                    ON related_publications.family_id = main_publications.family_id
                WHERE main_publications.application_kind = 'W'
                AND EXISTS (SELECT 1 FROM UNNEST(main_publications.cpc) AS cpc WHERE REGEXP_CONTAINS(cpc.code, "{BIOMEDICAL_IPC_CODE_PREFIX_RE}"))
            )
        )
        SELECT *
        FROM numbered_rows
        WHERE row_number = 1
    """
    await client.create_from_select(query, table_id)


async def _copy_gpr_annotations():
    """
    Copy annotations from GPR to a local table
    (ONLY diseases, and only the first instance of the term for a given publication_number)

    **Requires BigQuery publications table in our workspace**
    """
    client = BQDatabaseClient()
    await client.delete_table(GPR_ANNOTATIONS_TABLE)

    # 3.7TB query
    # NOTE: mixed mins/maxes
    query = f"""
        SELECT a.publication_number,
        a.preferred_name as preferred_name,
        max(ocid) as ocid,
        'diseases' as domain,
        (array_agg(a.source order by a.character_offset_start asc))[0] as source,
        (array_agg(a.confidence order by a.character_offset_start asc))[0] as confidence,
        min(a.character_offset_start) as character_offset_start,
        min(a.character_offset_end) as character_offset_end,
        FROM
        `patents-public-data.google_patents_research.annotations` a,
        `{BQ_DATASET_ID}.publications` p
        WHERE a.publication_number = p.publication_number
        AND a.domain='diseases'
        AND confidence >= 0.65
        AND preferred_name not in ('disease', 'syndrome')
        AND character_offset_start < 10000 -- otherwise, probably not the main indication?
        group by a.publication_number, a.preferred_name
    """
    await client.create_from_select(query, GPR_ANNOTATIONS_TABLE)


async def _copy_gpr_publications():
    """
    Copy publications from GPR to a BigQuery table in our workspace

    **Requires BigQuery publications table in our workspace**
    """
    client = BQDatabaseClient()
    await client.delete_table(GPR_PUBLICATIONS_TABLE)

    query = f"""
        SELECT gpr_pubs.* FROM
        `patents-public-data.google_patents_research.publications` gpr_pubs,
        `{BQ_DATASET_ID}.publications` p
        WHERE p.publication_number = gpr_pubs.publication_number
    """
    await client.create_from_select(query, GPR_PUBLICATIONS_TABLE)


async def copy_patent_tables():
    """
    Copy tables from patents-public-data to a local BQ dataset

    Idempotent (as in, tables are deleted and recreated) but not atomic, and also expensive (from a BigQuery standpoint)
    """

    # copy publications and gpr publications table (order matters)
    await _copy_publications()

    # depend upon publications
    await _copy_gpr_publications()
    await _copy_gpr_annotations()
