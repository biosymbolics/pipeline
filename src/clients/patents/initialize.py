from system import init

init()


import logging

from constants import COMMON_ENTITY_NAMES, SYNONYM_MAP
from clients.low_level.big_query import (
    execute_bg_query,
    query_to_bg_table,
    BQ_DATASET_ID,
)


logging.basicConfig(level=logging.INFO)

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))


def __create_synonym_map():
    """
    Create a table of synonyms
    """
    create = "CREATE TABLE patents.synonym_map (term STRING, synonym STRING);"
    execute_bg_query(create)
    entries = [
        f"('{entry[0].lower()}', '{entry[1].lower()}')" for entry in SYNONYM_MAP.items()
    ]
    query = "INSERT INTO patents.synonym_map (synonym, term) VALUES " + ",".join(
        entries
    )
    execute_bg_query(query)


def __copy_gpr_publications():
    """
    Copy publications from GPR to a local table
    """
    query = (
        "SELECT * FROM `patents-public-data.google_patents_research.publications` "
        "WHERE EXISTS "
        f'(SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))'
    )
    query_to_bg_table(query, "gpr_publications")


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
    SUPPRESSED_DOMAINS = (
        "anatomy",
        "chemCompound",  # 961,573,847
        "chemClass",
        "chemGroup",
        "inorgmat",
        "methods",  # lots of useless stuff
        "nutrients",
        "nutrition",  # 109,587,438
        "toxicity",  # 6,902,999
        "natprod",  # 23,053,704
        "species",  # 179,305,306
        "substances",  # 1,712,732,614
    )
    query = (
        "SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations "
        f"JOIN `{BQ_DATASET_ID}.gpr_publications` AS local_publications "
        "ON local_publications.publication_number = annotations.publication_number "
        "WHERE annotations.confidence > 0.69 "
        f"AND preferred_name not in {COMMON_ENTITY_NAMES} "
        f"AND domain not in {SUPPRESSED_DOMAINS} "
    )
    query_to_bg_table(query, "gpr_annotations")


def __copy_publications():
    """
    Copy publications from patents-public-data to a local table
    """
    query = (
        "SELECT publications.* FROM `patents-public-data.patents.publications` as publications "
        f"JOIN `{BQ_DATASET_ID}.gpr_publications` AS local_gpr "
        "ON local_gpr.publication_number = publications.publication_number "
        "WHERE application_kind = 'W' "
    )
    query_to_bg_table(query, "publications")


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


def __create_query_tables():
    """
    Create tables for use in app queries
    """
    logging.info("Creating patent tables for use in app queries")
    applications = (
        "SELECT "
        f"{','.join(FIELDS)} "
        f"FROM `{BQ_DATASET_ID}.publications` as pubs, "
        f"`{BQ_DATASET_ID}.gpr_publications` as gpr_pubs "
        "WHERE pubs.publication_number = gpr_pubs.publication_number "
    )
    entity_query = (
        "WITH ranked_terms AS ( "
        "SELECT "
        "publication_number,  ocid, "
        "LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) as term, "
        "domain, confidence, source, character_offset_start, "
        "ROW_NUMBER() OVER(PARTITION BY publication_number, LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) ORDER BY character_offset_start) as rank "
        f"FROM `{BQ_DATASET_ID}.gpr_annotations` a "
        f"LEFT JOIN `{BQ_DATASET_ID}.synonym_map` map ON LOWER(a.preferred_name) = map.synonym "
        ") "
        "SELECT "
        "publication_number, "
        "ARRAY_AGG( "
        "struct(ocid, term, domain, confidence, source, character_offset_start) "
        "ORDER BY character_offset_start "
        ") as annotations "
        "FROM ranked_terms "
        "WHERE rank = 1 "
        "GROUP BY publication_number "
    )
    query_to_bg_table(applications, "applications")
    query_to_bg_table(entity_query, "entities")


def main():
    # copy gpr_publications table
    __copy_gpr_publications()

    # copy publications table
    __copy_publications()

    # copy gpr_annotations table
    __copy_gpr_annotations()  # depends on publications

    # create synonym_map table (for final entity names)
    __create_synonym_map()

    # create the (small) tables against which the app will query
    __create_query_tables()


if __name__ == "__main__":
    main()
