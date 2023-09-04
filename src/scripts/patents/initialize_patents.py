"""
Functions to initialize the patents database
"""
import logging
import sys

from system import initialize

initialize()

from clients.low_level.big_query import BQDatabaseClient
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    AGGREGATED_ANNOTATIONS_TABLE,
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from scripts.patents.psql.copy_approvals import copy_patent_approvals
from scripts.patents.bq.copy_tables import copy_patent_tables
from scripts.patents.bq_to_psql import copy_bq_to_psql
from scripts.patents.psql.terms import create_patent_terms

from ._constants import APPLICATIONS_TABLE, TEXT_FIELDS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logger.info("Create a table of annotations for use in app queries")
    annotations_table = "annotations"

    client = PsqlDatabaseClient()
    client.delete_table(annotations_table, is_cascade=True)

    entity_query = f"""
        WITH terms AS (
                --- assignees as annotations
                SELECT
                    publication_number,
                    LOWER(case when map.term is null then assignee else map.term end) as term,
                    'assignees' as domain,
                    'record' as source,
                    1 as character_offset_start,
                    1 as character_offset_end
                FROM applications a,
                unnest(a.assignees) as assignee
                LEFT JOIN synonym_map map ON LOWER(assignee) = map.synonym

                UNION ALL

                --- inventors as annotations
                SELECT
                    publication_number,
                    LOWER(case when map.term is null then inventor else map.term end) as term,
                    'inventors' as domain,
                    'record' as source,
                    1 as character_offset_start,
                    1 as character_offset_end
                FROM applications a,
                unnest(a.inventors) as inventor
                LEFT JOIN synonym_map map ON LOWER(inventor) = map.synonym

                UNION ALL

                --- biosym (our) annotations
                SELECT
                    publication_number,
                    LOWER(case when syn_map.term is null then ba.term else syn_map.term end) as term,
                    domain,
                    source,
                    character_offset_start,
                    character_offset_end
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE} ba
                LEFT JOIN synonym_map syn_map ON LOWER(ba.term) = syn_map.synonym
                WHERE length(ba.term) > 0
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
    client.create_from_select(entity_query, annotations_table)

    client.create_indices(
        [
            {
                "table": annotations_table,
                "column": "publication_number",
            },
            {
                "table": annotations_table,
                "column": "term",
                "is_tgrm": True,
            },
        ]
    )

    mat_view_query = f"""
        DROP MATERIALIZED VIEW IF EXISTS {AGGREGATED_ANNOTATIONS_TABLE};
        CREATE MATERIALIZED VIEW {AGGREGATED_ANNOTATIONS_TABLE} AS
        SELECT
            publication_number,
            ARRAY_AGG(term) AS terms,
            ARRAY_AGG(domain) AS domains
        FROM {annotations_table}
        GROUP BY publication_number;
    """
    client.execute_query(mat_view_query)
    client.create_index(
        {
            "table": AGGREGATED_ANNOTATIONS_TABLE,
            "column": "publication_number",
        }
    )


def __create_biosym_annotations_source_table():
    """
    Creates biosym annotations table if need be
    (NOTE: does not check schema)
    """
    client = BQDatabaseClient()

    schema = {
        "publication_number": "STRING",
        "term": "STRING",
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
    logger.info(f"(Maybe) created table {SOURCE_BIOSYM_ANNOTATIONS_TABLE}")


def create_funcs():
    re_escape_sql = r"""
        CREATE OR REPLACE FUNCTION escape_regex_chars(text)
        RETURNS text
        LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
        $func$
        SELECT regexp_replace($1, '([!$()*+.:<=>?[\\\]^{|}-])', '\\\1', 'g')
        $func$;
    """
    PsqlDatabaseClient().execute_query(re_escape_sql)


def add_application_search():
    client = PsqlDatabaseClient()
    vector_sql = ("|| ' ' ||").join([f"coalesce({tf}, '')" for tf in TEXT_FIELDS])
    client.execute_query(
        f"""
            -- consider GENERATED ALWAYS AS/STORED
            ALTER TABLE applications ADD COLUMN text_search tsvector;
            UPDATE applications SET text_search = to_tsvector('english', {vector_sql});
        """,
        ignore_error=True,
    )
    client.create_index(
        {
            "table": APPLICATIONS_TABLE,
            "column": "text_search",
            "is_gin": True,
            "is_lower": False,
        }
    )


def main(bootstrap: bool = False):
    """
        Copy tables from patents-public-data to a local dataset.
        Order matters.

        Usage:
            >>> python3 -m scripts.patents.initialize_patents -bootstrap
            >>> python3 -m scripts.patents.initialize_patents

        Followed by:
        ```
        # from local machine
        pg_dump --no-owner patents > patents.psql
        zip patents.psql.zip patents.psql
        aws s3 mv s3://biosympatentsdb/patents.psql.zip s3://biosympatentsdb/patents.psql.zip.back-$(date +%Y-%m-%d)
        aws s3 cp patents.psql.zip s3://biosympatentsdb/patents.psql.zip
        rm patents.psql*

        # then proceeding in ec2
        aws configure sso
        aws s3 cp s3://biosympatentsdb/patents.psql.zip patents.psql.zip
        unzip patents.psql.zip
        export PASSWORD=$(aws ssm get-parameter --name /biosymbolics/pipeline/database/patents/main_password --with-decryption --query Parameter.Value --output text)
        echo "
    CREATE ROLE readaccess;
    GRANT USAGE ON SCHEMA public TO readaccess;
    GRANT SELECT ON ALL TABLES IN SCHEMA public TO readaccess;
    GRANT CONNECT ON DATABASE patents TO readaccess;
    GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readaccess;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readaccess;
    CREATE USER patents with password '$PASSWORD';
    GRANT readaccess TO patents;

    -- analyze annotations;
    -- analyze applications;
    -- reindex database patents;
        " >> patents.psql
    # pg_restore --clean -d patents -h 172.31.55.68 -p 5432 --username postgres --password patents.psql
    dropdb patents --force  -h 172.31.55.68 -p 5432 --username postgres
    createdb patents -h 172.31.55.68 -p 5432 --username postgres
    psql -d patents -h 172.31.55.68 -p 5432 --username postgres --password -f patents.psql
    rm patents.psql*
    ```
    """
    if bootstrap:
        # bigquery
        # copy gpr_publications, publications, gpr_annotations tables
        # idempotent but expensive
        create_funcs()
        __create_biosym_annotations_source_table()
        copy_patent_tables()
        # create patent applications etc in postgres
        copy_bq_to_psql()
        # copy data about approvals
        copy_patent_approvals()

        # adds column & index for application search
        add_application_search()

    # create patent terms (psql)
    create_patent_terms()

    # create annotations (psql)
    __create_annotations_table()

    # post
    # update terms set term=regexp_replace(term, ' gene$', '') where term ~* '^[a-z0-9-]{3,} gene$';
    # update terms set term= regexp_replace(term, '[.,]+$', '')  where term ~ '.*[ a-z][.,]+$';
    # update terms set term=regexp_replace(term, '(?i) (?:\[EPC\]|\[MoA\]|\(disposition\)|\(antigen\)|\(disease\)|\(disorder\)|\(finding\)|\(treatment\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$', '') where term ~* '(?:\[EPC\]|\[MoA\]|\(disposition\)|\(disease\)|\(treatment\)|\(antigen\)|\(disorder\)|\(finding\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$'
    # update terms set term=regexp_replace(term, '(?i)(agonist|inhibitor|blocker|modulator)s$', '\1') where term ~* '(agonist|inhibitor|blocker|modulator)s$';
    # update terms set term=regexp_replace(term, '(?i)(.*), A .*', '\1') where term ~* ', A ';
    # update terms set term=regexp_replace(term, '(?i)(.*), an?d? .*', '\1') where term ~* ', an?d? ';
    # update terms set term=regexp_replace(term, '^(?:\.\s?|\,\s?|;\s?|to[ ,]|for[ ,]|or[ ,]|the[ ,]|in[ ,]|are[ ,]|and[ ,]|as[ ,]|used[ ,]|its[ ,]|be[ ,])+', '') where term ~ '^(?:\.\s?|\,\s?|;\s?|to[ ,]|for[ ,]|or[ ,]|the[ ,]|in[ ,]|are[ ,]|and[ ,]|as[ ,]|used[ ,]|its[ ,]|be[ ,])+'
    # update terms set term=regexp_replace(term, '(.*)[ ]?(?:\(.*)', '\1') where term like '%(%' and term not like '%)%' and not term ~ '(?:\[|\])' and term not like '%-%-%'
    # update terms set term=regexp_replace(term, '(.*)[ ]?(?:.*\))', '\1') where term like '%)%' and term not like '%(%' and not term ~ '(?:\[|\])' and term not like '%-%-%'
    # update terms set term=regexp_replace(term, '\( \)', '') where term ~ '\( \)';


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.initialize_patents [-bootstrap]\nCopies and transforms patent data"
        )
        sys.exit()

    bootstrap: bool = "bootstrap" in sys.argv
    main(bootstrap)
