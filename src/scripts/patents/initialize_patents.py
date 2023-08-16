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
        WITH ranked_terms AS (
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
                FROM applications` a,
                unnest(a.assignee_harmonized) as assignee
                LEFT JOIN synonym_map map ON LOWER(assignee) = map.synonym

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
                FROM applications a,
                unnest(a.inventor_harmonized) as inventor
                LEFT JOIN synonym_map map ON LOWER(inventor) = map.synonym

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
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE} ba
                LEFT JOIN synonym_map syn_map ON LOWER(COALESCE(NULLIF(a.canonical_term, ''), a.original_term)) = syn_map.synonym
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
        __create_biosym_annotations_source_table()
        copy_patent_tables()

    # create patent applications etc in postgres
    # copy_bq_to_psql()

    # copy data about approvals
    copy_patent_approvals()

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
