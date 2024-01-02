"""
Functions to initialize the patents database
"""
import asyncio
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
from scripts.trials.copy_trials import copy_ctgov
from scripts.umls.copy_umls import copy_umls

from .constants import (
    GPR_ANNOTATIONS_TABLE,
    TEXT_FIELDS,
)
from .prep_bq_patents import copy_patent_tables
from .import_bq_patents import copy_bq_to_psql
from .terms import create_patent_terms

from ..approvals.copy_approvals import copy_approvals

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
            coalesce(max(t.instance_rollup), s.term) as instance_rollup,
            coalesce(max(t.category_rollup), s.term) as category_rollup
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
            coalesce(max(t.instance_rollup), s.term) as instance_rollup,
            coalesce(max(t.category_rollup), s.term) as category_rollup
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

        CREATE OR REPLACE FUNCTION zip(anyarray, anyarray)
        RETURNS SETOF anyarray LANGUAGE SQL AS
        $func$
        SELECT ARRAY[a,b] FROM (SELECT unnest($1) AS a, unnest($2) AS b) x;
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


async def main(bootstrap: bool = False):
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
    create extension vector; -- todo: move to beginning

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
        await copy_approvals()
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
    # update annotations set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) protein$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} protein$';
    # update annotations set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) protein, [a-z]{3,}$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} protein, [a-z]{3,}$';
    # update annotations set term=regexp_replace(term, '(?i)(?:\[EPC\]|\[MoA\]|\(disposition\)|\(antigen\)|\(disease\)|\(disorder\)|\(finding\)|\(treatment\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$', '', 'i') where term ~* '(?:\[EPC\]|\[MoA\]|\(disposition\)|\(disease\)|\(treatment\)|\(antigen\)|\(disorder\)|\(finding\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$';

    # update terms set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) gene$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} gene$';
    # update terms set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) protein$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} protein$';
    # update terms set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) protein, [a-z]{3,}$', '\1', 'i') where term ~* '^[a-z0-9-]{3,} protein, [a-z]{3,}$';
    # update terms set term=regexp_replace(term, '(?i)(?:\[EPC\]|\[MoA\]|\(disposition\)|\(antigen\)|\(disease\)|\(disorder\)|\(finding\)|\(treatment\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$', '', 'i') where term ~* '(?:\[EPC\]|\[MoA\]|\(disposition\)|\(disease\)|\(treatment\)|\(antigen\)|\(disorder\)|\(finding\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$';

    # update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)^([a-z0-9-]{3,}) gene$', '\1', 'i') where instance_rollup ~* '^[a-z0-9-]{3,} gene$';
    # update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)^([a-z0-9-]{3,}) protein$', '\1', 'i') where instance_rollup ~* '^[a-z0-9-]{3,} protein$';
    # update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)^([a-z0-9-]{3,}) protein, [a-z]{3,}$', '\1', 'i') where instance_rollup ~* '^[a-z0-9-]{3,} protein, [a-z]{3,}$';
    # update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)(?:\[EPC\]|\[MoA\]|\(disposition\)|\(antigen\)|\(disease\)|\(disorder\)|\(finding\)|\(treatment\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$', '', 'i') where instance_rollup ~* '(?:\[EPC\]|\[MoA\]|\(disposition\)|\(disease\)|\(treatment\)|\(antigen\)|\(disorder\)|\(finding\)|\(qualifier value\)|\(morphologic abnormality\)|\(procedure\)|\(product\)|\(substance\)|\(biomedical material\)|\(Chemistry\))$';

    # select t.term, ul.type_names, array_agg(distinct domain), array_agg(distinct original_term), count(*) from biosym_annotations, terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'mechanisms'
    #     and domain in ('compounds', 'devices', 'behavioral_interventions', 'procedures', 'diseases', 'diagnostics')
    #     and ul.type_ids::text[] <@ ARRAY['T120', 'T123', 'T195']
    #     and not original_term ~* '.*(?:diagnost).*'
    #     and lower(original_term)=ANY(t.synonyms)
    #     and array_length(t.synonyms, 1) < 50
    #     group by t.term, ul.type_names
    #     order by count(*) desc limit 100;

    # update biosym_annotations set domain='compounds'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'compounds'
    #     and domain in ('mechanisms', 'biologics', 'devices', 'behavioral_interventions', 'procedures', 'diseases', 'diagnostics')
    #     and ul.type_ids::text[] <@ ARRAY['T103', 'T104', 'T109',  'T127',  'T197', 'T200']
    #     and lower(original_term)=ANY(t.synonyms)
    #     and not original_term ~* '.*(?:inhibit|agoni|modulat)s?.*'
    #     and t.id not in ('C0268275', 'C1173729', 'C0031516', 'C0332837', 'C0037188', 'C0027627', 'C0011175', 'C0015967', 'C0151747', 'C0026837', 'C0700198', 'C0233656', 'C0043242', 'C0332875', 'C1510411', 'C2926602', 'C0242781', 'C0220811', 'C4074771', 'C0158328', 'C0011119', 'C0555975', 'C0877578', 'C0856151', 'C0263557', 'C0276640', 'C0858714', 'C0595920', 'C1318484', 'C0020488', 'C0278134', 'C0220724', 'C2029593', 'C0265604', 'C0012359', 'C0234985', 'C0027960', 'C1384489', 'C0277825', 'C0392483', 'C0010957', 'C0015376', 'C0011389', 'C0597561', 'C0036974', 'C0233494', 'C0011334', 'C0013146', 'C0030201', 'C0000925', 'C0332157', 'C0151908', 'C0024524', 'C0037293', 'C0233601', 'C4023747', 'C0262568', 'C0542351', 'C0036572', 'C0858950', 'C0001511', 'C0080194', 'C1514893', 'C0003516', 'C0332568', 'C0445243', 'C0349506', 'C0599156', 'C0033119', 'C4721411', 'C3658343', 'C1136365', 'C1704681', 'C0017374', 'C1334103', 'C0017345', 'C0017343', 'C0678941')
    #     and array_length(t.synonyms, 1) < 50;

    # update biosym_annotations set domain='biologics'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'biologics'
    #     and domain in ('compounds', 'devices', 'behavioral_interventions', 'procedures', 'diseases')
    #     and ul.type_ids::text[] <@ ARRAY['T038', 'T044', 'T045', 'T028','T043','T085','T086','T087','T088', 'T114', 'T116', 'T123', 'T125', 'T126', 'T129', 'T192']
    #     and lower(original_term)=ANY(t.synonyms)
    #     and not original_term ~* '.*(?:disease|disorder|syndrome)s?.*'
    #     and t.id not in ('C0268275', 'C1173729', 'C0031516', 'C0332837', 'C0037188', 'C0027627', 'C0011175', 'C0015967', 'C0151747', 'C0026837', 'C0700198', 'C0233656', 'C0043242', 'C0332875', 'C1510411', 'C2926602', 'C0242781', 'C0220811', 'C4074771', 'C0158328', 'C0011119', 'C0555975', 'C0877578', 'C0856151', 'C0263557', 'C0276640', 'C0858714', 'C0595920', 'C1318484', 'C0020488', 'C0278134', 'C0220724', 'C2029593', 'C0265604', 'C0012359', 'C0234985', 'C0027960', 'C1384489', 'C0277825', 'C0392483', 'C0010957', 'C0015376', 'C0011389', 'C0597561', 'C0036974', 'C0233494', 'C0011334', 'C0013146', 'C0030201', 'C0000925', 'C0332157', 'C0151908', 'C0024524', 'C0037293', 'C0233601', 'C4023747', 'C0262568', 'C0542351', 'C0036572', 'C0858950', 'C0001511', 'C0080194', 'C1514893', 'C0003516', 'C0332568', 'C0445243', 'C0349506', 'C0599156', 'C0033119', 'C4721411', 'C3658343', 'C1136365', 'C1704681', 'C0017374', 'C1334103', 'C0017345', 'C0017343', 'C0678941')
    #     and array_length(t.synonyms, 1) < 20;

    # update biosym_annotations set domain='devices'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'devices'
    #     and domain in ('compounds', 'mechanisms', 'biologics', 'behavioral_interventions', 'procedures', 'diseases')
    #     and ul.type_ids::text[] <@ ARRAY['T074', 'T075', 'T203']
    #     and lower(original_term)=ANY(t.synonyms);

    # update biosym_annotations set domain='procedures'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'procedures'
    #     and domain in ('compounds', 'mechanisms', 'biologics', 'behavioral_interventions', 'devices', 'diseases')
    #     and ul.type_ids::text[] <@ ARRAY['T059', 'T061']
    #     and lower(original_term)=ANY(t.synonyms);

    # update biosym_annotations set domain='diagnostics'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'diagnostics'
    #     and domain in ('compounds', 'mechanisms', 'biologics', 'behavioral_interventions', 'procedures', 'diseases', 'devices')
    #     and ul.type_ids::text[] <@ ARRAY['T060', 'T034', 'T130']
    #     and lower(original_term)=ANY(t.synonyms);

    # update biosym_annotations set domain='diseases'
    #     from terms t, umls_lookup ul
    #     where ul.id=t.id
    #     and domain<>'diseases'
    #     and domain in ('compounds', 'mechanisms', 'biologics', 'behavioral_interventions', 'devices', 'procedures')
    #     and ul.type_ids::text[] <@ ARRAY['T019', 'T020', 'T037', 'T046', 'T047', 'T048', 'T184', 'T190', 'T191']
    #     and lower(original_term)=ANY(t.synonyms)
    #     and not original_term ~* '.*(?:anti|inflammation|therapeutic|repair).*'
    #     and t.id not in ('C0268275', 'C1173729', 'C0031516', 'C0332837', 'C0037188', 'C0027627', 'C0011175', 'C0015967', 'C0151747', 'C0026837', 'C0700198', 'C0233656', 'C0043242', 'C0332875', 'C1510411', 'C2926602', 'C0242781', 'C0220811', 'C4074771', 'C0158328', 'C0011119', 'C0555975', 'C0877578', 'C0856151', 'C0263557', 'C0276640', 'C0858714', 'C0595920', 'C1318484', 'C0020488', 'C0278134', 'C0220724', 'C2029593', 'C0265604', 'C0012359', 'C0234985', 'C0027960', 'C1384489', 'C0277825', 'C0392483', 'C0010957', 'C0015376', 'C0011389', 'C0597561', 'C0036974', 'C0233494', 'C0011334', 'C0013146', 'C0030201', 'C0000925', 'C0332157', 'C0151908', 'C0024524', 'C0037293', 'C0233601', 'C4023747', 'C0262568', 'C0542351', 'C0036572', 'C0858950', 'C0001511', 'C0080194', 'C1514893', 'C0003516', 'C0332568', 'C0445243', 'C0349506', 'C0599156', 'C0033119', 'C4721411', 'C3658343', 'C1136365', 'C1704681', 'C0017374', 'C1334103', 'C0017345', 'C0017343', 'C0678941')
    #     and array_length(t.synonyms, 1) < 15;

    # UPDATE trials
    # SET interventions = sub.new_interventions
    # FROM (
    #   SELECT nct_id, array_agg(i) as new_interventions
    #   FROM trials, unnest(interventions) i
    #   where not i ~* '\ystandard therapy\y'
    #   group by nct_id
    # ) sub
    # where sub.nct_id=trials.nct_id

    # UPDATE trials
    # SET interventions = sub.new_interventions
    # FROM (
    #   SELECT nct_id, array_agg(i), array_agg(
    #       regexp_replace(i, '(.*) [0-9]+%', '\1', 'ig')
    #   ) AS new_interventions
    #   FROM trials, unnest(interventions) i
    #   where array_to_string(interventions, '::') ~* ' [0-9]+%'
    #   group by nct_id
    # ) sub
    # where sub.nct_id=trials.nct_id

    # UPDATE trials
    # SET interventions = sub.new_interventions
    # FROM (
    #   SELECT nct_id, array_agg(i), array_agg(
    #       trim(i)
    #   ) AS new_interventions
    #   FROM trials, unnest(interventions) i
    #   group by nct_id
    # ) sub
    # where sub.nct_id=trials.nct_id

    # update annotations set instance_rollup=lower(instance_rollup) where instance_rollup<>lower(instance_rollup);
    # update annotations set instance_rollup=term where domain not in ('attributes', 'assignees') and lower(instance_rollup) in  ('pharmacologic substance', 'inhibitor', '11-dehydrocorticosterone', 'peptides', 'disease', 'compound', 'base sequence','salts', 'bacteria', 'alleles', 'modulator', 'syndrome', 'antibodies', 'abnormality of digestive system morphology', 'antagonist', 'agonist', 'proteins', 'catalyst', 'agent', 'promoters', 'enhancer of transcription', 'laboratory chemicals', 'monomers', 'syndrome', 'disease', 'cells', 'chemical group', 'copolymer', 'polymers', 'plant-based natural product', 'degrader')
    # delete from annotations where lower(term) in ('inhibitor', '11-dehydrocorticosterone', 'composition', 'composition comprising', 'peptides', 'disease', 'compound', 'base sequence','salts', 'bacteria', 'alleles', 'modulator', 'syndrome', 'antibodies', 'abnormality of digestive system morphology', 'antagonist', 'agonist', 'proteins', 'catalyst', 'agent', 'promoters', 'enhancer of transcription', 'laboratory chemicals', 'monomers', 'syndrome', 'disease', 'cells', 'chemical group', 'copolymer', 'polymers', 'plant-based natural product', 'degrader');
    # refresh materialized view aggregated_annotations;

    # update annotations set domain='diseases' where domain<>'diseases' and term ~* '^.* leukemias?$';
    # update biosym_annotations_source set domain='diseases' where domain<>'diseases' and term ~* '^.* leukemias?$';
    # update biosym_annotations set domain='diseases' where domain<>'diseases' and original_term ~* '^.* leukemias?$';


#     update annotations set domain='biologics' where term in ('dna', )
#     update annotations set domain='biologics' where term ilike '% channel' and domain='compounds';
#     update annotations set domain='biologics' where term ilike '% cell line' and domain='devices';
#     update annotations set domain='compounds' where term ilike '% fumarate' and domain<>'compounds';
#     delete from annotations where term='negation efficacious';
#     update annotations set domain='biologics' where term ilike '%oligonucleotides' and domain<>'biologics';

#     update annotations set term=regexp_replace(term, '(?i)^(.*) combinations$', '\1', 'i') where term ~* '^.* combinations$';
#     update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)^(.*) combinations$', '\1', 'i') where instance_rollup ~* '^.* combinations$';
# delete from annotations where term in ('dosage forms bead', 'dosage froms matrix', 'composition comprising said');

#     update annotations set term=regexp_replace(term, '(?i)\ygene (agonist|antagonist|inhibitor)\y', '\1', 'i') where term ~* 'gene (agonist|antagonist|inhibitor)';
#     update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)\ygene (agonist|antagonist|inhibitor)\y', '\1', 'i') where instance_rollup ~* 'gene (agonist|antagonist|inhibitor)';

#     update annotations set instance_rollup=regexp_replace(instance_rollup, '(?i)qualitative form', 'form', 'i') where instance_rollup ~* 'qualitative form';
#     update annotations set term=regexp_replace(term, 'group specimen', 'group', 'i') where term ~* 'group specimen';
#     update annotations set instance_rollup=regexp_replace(instance_rollup, 'group specimen', 'group', 'i') where instance_rollup ~* 'group specimen';

#     update annotations set term=regexp_replace(term, '\ygenerating\y', '', 'i') where term ~* '\ygenerating\y';

#     update annotations set term=regexp_replace(term, '[ ]{2,}', ' ', 'g') where term ~* '[ ]{2,}';
#     update annotations set term=trim(term) where trim(term) <> term;

#  update annotations set term=regexp_replace(term, '^(activator|inhibitor|agonist|antagonist) (.*) receptor', '\2 \1', 'g') where term ~* '^(activator|inhibitor|agonist|antagonist) (.*) receptor'

#  update annotations set instance_rollup=regexp_replace(instance_rollup, '^(activator|inhibitor|agonist|antagonist) (.*) receptor', '\2 \1', 'g') where instance_rollup ~* '^(activator|inhibitor|agonist|antagonist) (.*) receptor'


# update annotations set instance_rollup=regexp_replace(instance_rollup, 'aberranta', 'aberrant', 'i') where instance_rollup ~* 'aberranta';
#     update annotations set instance_rollup=regexp_replace(instance_rollup, 'social interaction', 'interaction', 'i') where instance_rollup ~* 'social interaction';


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m scripts.patents.initialize_patents [-bootstrap]\nCopies and transforms patent data"
        )
        sys.exit()

    bootstrap: bool = "bootstrap" in sys.argv
    asyncio.run(main(bootstrap))
