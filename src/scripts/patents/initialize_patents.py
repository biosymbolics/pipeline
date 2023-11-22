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
    APPLICATIONS_TABLE,
    ANNOTATIONS_TABLE,
)
from scripts.ctgov.copy_ctgov import copy_ctgov
from scripts.umls.copy_umls import copy_umls

from .constants import (
    GPR_ANNOTATIONS_TABLE,
    TEXT_FIELDS,
)
from .copy_approvals import copy_approvals
from .prep_bq_patents import copy_patent_tables
from .import_bq_patents import copy_bq_to_psql
from .terms import create_patent_terms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logger.info("Create a table of annotations for use in app queries")

    client = PsqlDatabaseClient()
    # to delete materialized view
    client.delete_table(ANNOTATIONS_TABLE, is_cascade=True)

    annotations_query = f"""
        --- assignees as annotations
        SELECT
            publication_number,
            (CASE WHEN map.term is null THEN LOWER(assignee) ELSE map.term end) as term,
            null as id,
            'assignees' as domain,
            'record' as source,
            1 as character_offset_start,
            1 as character_offset_end,
            (CASE WHEN map.term is null THEN LOWER(assignee) ELSE map.term end) as instance_rollup,
            (CASE WHEN map.term is null THEN LOWER(assignee) ELSE map.term end) as category_rollup
        FROM applications a,
        unnest(a.assignees) as assignee
        LEFT JOIN synonym_map map ON LOWER(assignee) = map.synonym

        UNION ALL

        --- inventors as annotations
        SELECT
            publication_number,
            (CASE WHEN map.term is null THEN lower(inventor) ELSE map.term end) as term,
            null as id,
            'inventors' as domain,
            'record' as source,
            1 as character_offset_start,
            1 as character_offset_end,
            (CASE WHEN map.term is null THEN lower(inventor) ELSE map.term end) as instance_rollup,
            (CASE WHEN map.term is null THEN lower(inventor) ELSE map.term end) as category_rollup
        FROM applications a,
        unnest(a.inventors) as inventor
        LEFT JOIN synonym_map map ON LOWER(inventor) = map.synonym
    """
    client.create_from_select(annotations_query, ANNOTATIONS_TABLE)

    # add biosym annotations
    client.select_insert_into_table(
        f"""
        SELECT publication_number,
            s.term as term,
            s.id as id,
            domain,
            max(source) as source,
            min(character_offset_start) as character_offset_start,
            min(character_offset_end) as character_offset_end,
            max(t.instance_rollup) as instance_rollup,
            max(t.category_rollup) as category_rollup
        from (
            SELECT
                publication_number,
                (CASE WHEN map.term is null THEN lower(original_term) ELSE map.term end) as term,
                (CASE WHEN map.id is null THEN lower(original_term) ELSE map.id end) as id,
                domain,
                source,
                character_offset_start,
                character_offset_end
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
                LEFT JOIN synonym_map map ON LOWER(original_term) = map.synonym
            ) s
            LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
            group by publication_number, s.term, s.id, domain
        """,
        ANNOTATIONS_TABLE,
    )

    # add gpr annotations
    client.select_insert_into_table(
        f"""
        SELECT publication_number,
            s.term as term,
            s.id as id,
            domain,
            max(source) as source,
            min(character_offset_start) as character_offset_start,
            min(character_offset_end) as character_offset_end,
            max(t.instance_rollup) as instance_rollup,
            max(t.category_rollup) as category_rollup
        from (
            SELECT
                publication_number,
                (CASE WHEN map.term is null THEN lower(preferred_name) ELSE map.term end) as term,
                (CASE WHEN map.id is null THEN lower(preferred_name) ELSE map.id end) as id,
                domain,
                source,
                character_offset_start,
                character_offset_end
                FROM {GPR_ANNOTATIONS_TABLE}
                LEFT JOIN synonym_map map ON LOWER(preferred_name) = map.synonym
            ) s
            LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
            group by publication_number, s.term, s.id, domain
        """,
        ANNOTATIONS_TABLE,
    )

    # add attributes at the last moment
    client.select_insert_into_table(
        f"""
            SELECT
                publication_number,
                original_term as term,
                original_term as id,
                domain,
                source,
                character_offset_start,
                character_offset_end,
                original_term as instance_rollup,
                original_term as category_rollup
            from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
            where domain='attributes'
        """,
        ANNOTATIONS_TABLE,
    )
    client.create_indices(
        [
            {
                "table": ANNOTATIONS_TABLE,
                "column": "publication_number",
            },
            {
                "table": ANNOTATIONS_TABLE,
                "column": "term",
            },
            {
                "table": ANNOTATIONS_TABLE,
                "column": "id",
            },
        ]
    )

    # non-distinct because order matters
    mat_view_query = f"""
        DROP MATERIALIZED VIEW IF EXISTS {AGGREGATED_ANNOTATIONS_TABLE};
        CREATE MATERIALIZED VIEW {AGGREGATED_ANNOTATIONS_TABLE} AS
        SELECT
            publication_number,
            ARRAY_AGG(a.term) AS terms,
            ARRAY_AGG(domain) AS domains,
            ARRAY_AGG(instance_rollup) as instance_rollup,
            ARRAY_AGG(category_rollup) as category_rollup,
            ARRAY_CAT(ARRAY_AGG(instance_rollup), ARRAY_AGG(a.term)) as search_terms
        FROM {ANNOTATIONS_TABLE} a
        GROUP BY publication_number;
    """
    client.execute_query(mat_view_query)
    client.create_indices(
        [
            {
                "table": AGGREGATED_ANNOTATIONS_TABLE,
                "column": "publication_number",
            },
            {
                "table": AGGREGATED_ANNOTATIONS_TABLE,
                "column": "search_terms",
                "is_gin": True,
            },
        ]
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
            ALTER TABLE {APPLICATIONS_TABLE} ADD COLUMN text_search tsvector;
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
    pg_dump --no-owner patents \
        -t aggregated_annotations \
        -t annotations \
        -t applications \
        -t terms \
        -t patent_to_trial \
        -t trials \
        -t regulatory_approvals \
        -t umls_lookup \
        -t umls_graph \
        -t term_ids \
        -t companies \
        -t patent_clindev_predictions \
        -t patent_to_regulatory_approval > patents.psql
    zip patents.psql.zip patents.psql
    aws s3 mv s3://biosympatentsdb/patents.psql.zip s3://biosympatentsdb/patents.psql.zip.back-$(date +%Y-%m-%d)
    aws s3 cp patents.psql.zip s3://biosympatentsdb/patents.psql.zip
    rm patents.psql*

    # then proceeding in ec2
    aws configure sso
    aws s3 cp s3://biosympatentsdb/patents.psql.zip patents.psql.zip
    unzip patents.psql.zip
    y
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

    -- vacuum analyze;
    -- analyze applications;
    -- reindex database patents;
        " >> patents.psql
    echo $PASSWORD
    dropdb patents --force  -h 172.31.55.68 -p 5432 --username postgres
    createdb patents -h 172.31.55.68 -p 5432 --username postgres
    psql -d patents -h 172.31.55.68 -p 5432 --username postgres --password -f patents.psql
    rm patents.psql*
    ```

    if new bastion:
    - yum install postgresql15
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
        copy_approvals()
        # adds column & index for application search
        add_application_search()
        # UMLS records (slow due to betweenness centrality calc)
        copy_umls()

    # create patent terms (psql)
    create_patent_terms()

    # create annotations (psql)
    __create_annotations_table()

    # copy trial data
    copy_ctgov()

    # post
    # TODO: same mods to trials? or needs to be in-line adjustment in normalizing/mapping
    # update annotations set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) gene$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} gene$';
    # update annotations set term=regexp_replace(term, '(?i)(?:\[EPC\]|\[MoA\]|\(disposition\)|\(antigen\)|\(disease\)|\(disorder\)|\(finding\)|\(treatment\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$', '', 'i') where term ~* '(?:\[EPC\]|\[MoA\]|\(disposition\)|\(disease\)|\(treatment\)|\(antigen\)|\(disorder\)|\(finding\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$';

    # UPDATE trials
    # SET interventions = sub.new_interventions
    # FROM (
    #   SELECT nct_id, array_agg(i), array_agg(
    #       regexp_replace(i, '\y[0-9]{1,}\.?[0-9]*[ ]?(?:mg|mcg|ug|µg|ml|gm?)(?:[ /](?:kgs?|lbs?|m2|m^2|day|mg))*\y', '', 'ig')
    #   ) AS new_interventions
    #   FROM trials, unnest(interventions) i
    #   group by nct_id
    # ) sub
    # where sub.nct_id=trials.nct_id


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.initialize_patents [-bootstrap]\nCopies and transforms patent data"
        )
        sys.exit()

    bootstrap: bool = "bootstrap" in sys.argv
    main(bootstrap)
