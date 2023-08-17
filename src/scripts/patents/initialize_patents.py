"""
Functions to initialize the patents database
"""
import logging
import sys
from scripts.patents.psql.copy_approvals import copy_patent_approvals

from system import initialize

initialize()

from clients.low_level.big_query import BQDatabaseClient
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from scripts.patents.bq.copy_tables import copy_patent_tables
from scripts.patents.bq_to_psql import copy_bq_to_psql
from scripts.patents.psql.terms import create_patent_terms

logging.basicConfig(level=logging.INFO)


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logging.info("Create a table of annotations for use in app queries")
    table_name = "annotations"

    client = PsqlDatabaseClient()
    client.delete_table(table_name)

    entity_query = f"""
        WITH terms AS (
                --- assignees as annotations
                SELECT
                    publication_number,
                    LOWER(case when map.term is null then assignee else map.term end) as term,
                    'assignee' as domain,
                    'record' as source,
                    1 as character_offset_start,
                    1 as character_offset_end
                FROM applications a,
                unnest(a.assignee_harmonized) as assignee
                LEFT JOIN synonym_map map ON LOWER(assignee) = map.synonym

                UNION ALL

                --- inventors as annotations
                SELECT
                    publication_number,
                    LOWER(case when map.term is null then inventor else map.term end) as term,
                    'inventor' as domain,
                    'record' as source,
                    1 as character_offset_start,
                    1 as character_offset_end
                FROM applications a,
                unnest(a.inventor_harmonized) as inventor
                LEFT JOIN synonym_map map ON LOWER(inventor) = map.synonym

                UNION ALL

                --- biosym (our) annotations
                SELECT
                    publication_number,
                    LOWER(case when syn_map.term is null then ba.original_term else syn_map.term end) as term,
                    domain,
                    source,
                    character_offset_start,
                    character_offset_end
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE} ba
                LEFT JOIN synonym_map syn_map ON LOWER(ba.original_term) = syn_map.synonym
                WHERE length(ba.original_term) > 0
        )
        SELECT
            publication_number,
            term,
            domain,
            source,
            character_offset_start,
            character_offset_end
        FROM terms
        ORDER BY character_offset_start
    """
    client.select_to_table(entity_query, table_name)


def __create_biosym_annotations_source_table():
    """
    Creates biosym annotations table if need be
    (NOTE: does not check schema)
    """
    client = BQDatabaseClient()

    schema = {
        "publication_number": "STRING",
        "original_term": "STRING",
        "canonical_term": "STRING",
        "domain": "STRING",
        "confidence": "FLOAT",
        "source": "STRING",
        "character_offset_start": "INTEGER",
        "character_offset_end": "INTEGER",
    }
    client.create_table(
        SOURCE_BIOSYM_ANNOTATIONS_TABLE,
        schema,
        exists_ok=True,
        truncate_if_exists=False,
    )
    logging.info(f"(Maybe) created table {SOURCE_BIOSYM_ANNOTATIONS_TABLE}")


def create_funcs():
    sql = r"""
        CREATE OR REPLACE FUNCTION escape_regex_chars(text)
        RETURNS text
        LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
        $func$
        SELECT regexp_replace($1, '([!$()*+.:<=>?[\\\]^{|}-])', '\\\1', 'g')
        $func$;
    """
    PsqlDatabaseClient().execute_query(sql)


def main(bootstrap: bool = False):
    """
    Copy tables from patents-public-data to a local dataset.
    Order matters.

    Usage:
        >>> python3 -m scripts.patents.initialize_patents -bootstrap
        >>> python3 -m scripts.patents.initialize_patents
    """
    if bootstrap:
        # bigquery
        # copy gpr_publications, publications, gpr_annotations tables
        # idempotent but expensive
        create_funcs()
        __create_biosym_annotations_source_table()
        copy_patent_tables()

    # create patent applications etc in postgres
    # copy_bq_to_psql()

    # copy data about approvals
    # copy_patent_approvals()

    # create patent terms (psql)
    create_patent_terms()

    # create annotations (psql)
    __create_annotations_table()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.initialize_patents [-bootstrap]\nCopies and transforms patent data"
        )
        sys.exit()

    bootstrap: bool = "bootstrap" in sys.argv
    main(bootstrap)
