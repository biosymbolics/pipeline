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
from scripts.ctgov.copy_ctgov import copy_ctgov

from .constants import (
    APPLICATIONS_TABLE,
    ANNOTATIONS_TABLE,
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

    entity_query = f"""
        WITH terms AS (
                --- assignees as annotations
                SELECT
                    publication_number,
                    LOWER(case when map.term is null then assignee else map.term end) as term,
                    'assignees' as domain,
                    'record' as source,
                    1 as character_offset_start,
                    1 as character_offset_end,
                    '' as instance_ancestor,
                    '' as category_ancestor
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
                    1 as character_offset_end,
                    '' as instance_ancestor,
                    '' as category_ancestor
                FROM applications a,
                unnest(a.inventors) as inventor
                LEFT JOIN synonym_map map ON LOWER(inventor) = map.synonym

                UNION ALL

                -- gpr annotations (just diseases)
                SELECT
                    publication_number,
                    s.norm_term as term,
                    domain,
                    source,
                    character_offset_start,
                    character_offset_end,
                    '' as instance_ancestor,
                    '' as category_ancestor
                    FROM (
                        SELECT
                            *,
                            LOWER(case when map.term is null then preferred_name else map.term end) as norm_term,
                            ROW_NUMBER() OVER(
                                PARTITION BY publication_number,
                                (case when map.term is null then preferred_name else map.term end)
                                ORDER BY character_offset_start DESC
                            ) AS rn
                        FROM {GPR_ANNOTATIONS_TABLE}
                        LEFT JOIN synonym_map map ON LOWER(preferred_name) = map.synonym
                    ) s
                    WHERE rn = 1

                UNION ALL

                --- biosym annotations
                SELECT
                    publication_number,
                    s.norm_term as term,
                    domain,
                    source,
                    character_offset_start,
                    character_offset_end,
                    t.instance_ancestor as instance_ancestor,
                    t.category_ancestor as category_ancestor,
                    FROM (
                        SELECT
                            *,
                            LOWER(case when map.term is null then original_term else map.term end) as norm_term,
                            ROW_NUMBER() OVER(
                                PARTITION BY publication_number,
                                (case when map.term is null then original_term else map.term end),
                                domain
                                ORDER BY character_offset_start DESC
                            ) AS rn
                        FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE} ba
                        LEFT JOIN synonym_map map ON LOWER(original_term) = map.synonym
                        LEFT JOIN terms t on map.term = t.term
                        WHERE length(ba.term) > 0
                    ) s
                    WHERE rn = 1
        )
        SELECT
            publication_number,
            term,
            domain,
            source,
            character_offset_start,
            character_offset_end,
            instance_ancestor, -- max instance term (i.e. the furthest away ancestor still considered an "instance" entity)
            category_ancestor -- min category term (i.e. the closest ancestor considered to be a category)
        FROM terms
        ORDER BY character_offset_start
    """
    client.create_from_select(entity_query, ANNOTATIONS_TABLE)

    # add attributes at the last moment
    client.select_insert_into_table(
        f"""
            SELECT
                publication_number,
                original_term,
                domain,
                source,
                character_offset_start,
                character_offset_end,
                '' as instance_ancestor,
                '' as category_ancestor
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
        ]
    )

    # non-distinct because order matters
    mat_view_query = f"""
        DROP MATERIALIZED VIEW IF EXISTS {AGGREGATED_ANNOTATIONS_TABLE};
        CREATE MATERIALIZED VIEW {AGGREGATED_ANNOTATIONS_TABLE} AS
        SELECT
            publication_number,
            ARRAY_AGG(term) AS terms,
            ARRAY_AGG(domain) AS domains
        FROM {ANNOTATIONS_TABLE}
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
                "column": "terms",
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

    -- analyze annotations;
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
    # update annotations set term=regexp_replace(term, '(?i)(agonist|inhibitor|blocker|modulator)s$', '\1') where term ~* '(agonist|inhibitor|blocker|modulator)s$';
    # update annotations set term=regexp_replace(term, '(?i)^([a-z0-9-]{3,}) protein$', '\1', 'i') where  term ~* '^[a-z0-9]{3,5} protein$';
    # update annotations set term=regexp_replace(term, ' (?:(?:super)?family )?protein$', '') where  term ~* '^[a-z0-9]{3,5}[0-9] (?:(?:super)?family )?protein$';
    # update annotations set term=regexp_replace(term, '(?:target(?:ed|ing) antibody|antibody conjugate)', 'antibody') where term ~* '\y(?:target(?:ed|ing) antibody|antibody conjugate)\y';
    # update annotations set term=regexp_replace(term, ', rat', '', 'i') where term ~* ', rat$';
    # update annotations set term=regexp_replace(term, ', human', '', 'i') where term ~* ', human$';
    # update annotations set term=regexp_replace(term, 'modulat(?:ed?|ing|ion)$', 'modulator') where term ~* '\ymodulat(?:ed?|ing|ion)$';
    # update annotations set term=regexp_replace(term, 'activat(?:ed?|ing|ion)$', 'activator') where term ~* '\yactivat(?:ed?|ing|ion)$'; -- 0 recs
    # update annotations set term=regexp_replace(term, 'stimulat(?:ed?|ing|ion)$', 'stimulator') where term ~* '\ystimulat(?:ed?|ing|ion)$';
    # update annotations set term=regexp_replace(term, 'stabili[sz](?:ed?|ing|ion)$', 'stabilizer') where term ~* '\ystabili[sz](?:ed?|ing|ion)$';
    # update annotations set term=regexp_replace(term, 'inhibit(?:ion|ing|ed)$', 'inhibitor') where term ~* '\yinhibit(?:ion|ing|ed)$';
    # update annotations set term=regexp_replace(term, 'agonist(?:ic)? action$', 'agonist') where term ~* 'agonist(?:ic)? action$';
    # update annotations set term=regexp_replace(term, 'receptor activat(?:ion|or)$', 'activator') where term ~* '\yreceptor activat(?:ion|or)$';
    # update annotations set term=regexp_replace(term, 'agon(?:ism|i[zs])(?:ing|ed|e)$', 'agonist') where term ~* '\yagon(?:ism|i[zs])(?:ing|ed|e)$';
    # update annotations set term=regexp_replace(term, '^([a-z]{2,3}[0-9]{0,2}) ([a-zαβγδεζηθικλμνξοπρστυφχψω]{1}[ ])', '\1\2') where term ~* '^[a-z]{2,3}[0-9]{0,2} [a-zαβγδεζηθικλμνξοπρστυφχψω]{1} (?:inhibitor|modulator|antagonist|agonist|protein|(?:poly)?peptide|antibody|isoform|domain|bispecific|chain|activator|stimulator|dna)';
    # update annotations set term=regexp_replace(term, '(?:\.\s?|\,\s?|;\s?| to| for| or| the| in| are| and| as| used| using| its| be| which)+$', '', 'i') where term ~* '(?:\.\s?|\,\s?|;\s?| to| for| or| the| in| are| and| as| used| using| its| be| which)+$';
    # update annotations set term=regexp_replace(term, '^vitro', 'in-vitro', 'i') where term ~* '^vitro .*';
    # update annotations set term=regexp_replace(term, '^vivo', 'in-vivo', 'i') where term ~* '^vivo .*';
    # update annotations set term=regexp_replace(term, '^(?:\.\s?|\,\s?|;\s?|to[ ,]|tha(?:n|t)[ ,]|for[ ,]|or[ ,]|then?[ ,]|are[ ,]|and[ ,]|as[ ,]|used[ ,]|using[ ,]|its[ ,]|be[ ,])+', '', 'i') where term ~* '^(?:\.\s?|\,\s?|;\s?|to[ ,]|for[ ,]|or[ ,]|then?[ ,]|are[ ,]|tha(?:n|t)[ ,]|and[ ,]|as[ ,]|used[ ,]|using[ ,]|its[ ,]|be[ ,])+';
    # update annotations set term=regexp_replace(term, '(.*)[ ]?(?:\(.*)', '\1') where term like '%(%' and term not like '%)%' and not term ~ '(?:\[|\])' and term not like '%-%-%';
    # update annotations set term=regexp_replace(term, '(.*)[ ]?(?:.*\))', '\1') where term like '%)%' and term not like '%(%' and not term ~ '(?:\[|\])' and term not like '%-%-%';
    # update annotations set term=regexp_replace(term, '[0-9a-z]{0,1}\)[ ](.*)', '\1', 'i') where term ~* '^[0-9a-z]{0,1}\)[ ]' and not term ~ '(?:\[|\])' and term not like '%-%-%';
    # update annotations set term=regexp_replace(term, '\( \)', '') where term ~ '\( \)';
    # update annotations set term=regexp_replace(term, '[.,]+$', '')  where term ~ '.*[ a-z][.,]+$';
    # update annotations set term=regexp_replace(term, '\s{2,}', ' ', 'g') where term ~* '\s{2,}';
    # update annotations set term=regexp_replace(term, '\s{1,}$', '', 'g') where term ~* '\s{1,}$';
    # update annotations set term=regexp_replace(term, '^((?:ant)?agonists?) ([^ ]+)$', '\2 \1') where term ~* '^(?:ant)?agonist [^ ]+$';
    # update annotations set term=regexp_replace(term, '^(inhibit[a-z]+) ([^ ]+)$', '\2 \1') where term ~* '^inhibit[a-z]+ [^ ]+$';
    # update annotations set term=regexp_replace(term, '^(modulat[a-z]+) ([^ ]+)$', '\2 \1') where term ~* '^modulat(?:or|ion|ing|ed)s? [^ ]+$';

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
