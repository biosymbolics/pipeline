from google.cloud import bigquery
import logging

from constants import COMMON_ENTITY_NAMES, SYNONYM_MAP

logging.basicConfig(level=logging.INFO)

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))

PROJECT = "fair-abbey-386416"
DATASET = "patents"
DATASET_ID = PROJECT + "." + DATASET


def __run_bq_query(query: str):
    """
    Query bigquery

    Args:
        query (str): SQL query
    """
    # Create a client
    client = bigquery.Client()

    logging.info("Starting query: %s", query)
    query_job = client.query(query)

    # Wait for the job to complete
    query_job.result()
    logging.info("Query complete")


def query_to_table(query, new_table_name: str):
    """
    Create a new table from a query
    """
    create_table_query = f"CREATE TABLE `{DATASET_ID}.{new_table_name}` AS {query};"
    __run_bq_query(create_table_query)


def create_synonym_map():
    """
    Create a table of synonyms
    """
    create = "CREATE TABLE patents.synonym_map (term STRING, synonym STRING);"
    __run_bq_query(create)
    entries = [
        f"('{entry[0].lower()}', '{entry[1].lower()}')" for entry in SYNONYM_MAP.items()
    ]
    query = "INSERT INTO patents.synonym_map (synonym, term) VALUES " + ",".join(
        entries
    )
    __run_bq_query(query)


def copy_gpr_publications():
    """
    Copy publications from GPR to a local table
    """
    query = (
        "SELECT * FROM `patents-public-data.google_patents_research.publications` "
        "WHERE EXISTS "
        f'(SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))'
    )
    query_to_table(query, "gpr_publications")


def copy_gpr_annotations():
    """
    Copy annotations from GPR to a local table
    """
    SUPPRESSED_DOMAINS = (
        "chemCompound",  # 961,573,847
        "inorgmat",
        "nutrients",
        "nutrition",  # 109,587,438
        "toxicity",  # 6,902,999
        "natprod",  # 23,053,704
        "species",  # 179,305,306
        "substances",  # 1,712,732,614
    )
    query = (
        "SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations "
        f"JOIN `{DATASET_ID}.gpr_publications` AS local_publications "
        "ON local_publications.publication_number = annotations.publication_number "
        "WHERE annotations.confidence > 0.69 "
        f"AND preferred_name not in {COMMON_ENTITY_NAMES} "
        f"AND domain not in {SUPPRESSED_DOMAINS} "
    )
    query_to_table(query, "gpr_annotations")


def copy_publications():
    """
    Copy publications from patents-public-data to a local table
    """
    query = (
        "SELECT publications.* FROM `patents-public-data.patents.publications` as publications "
        f"JOIN `{DATASET_ID}.gpr_publications` AS local_gpr "
        "ON local_gpr.publication_number = publications.publication_number "
        "WHERE application_kind = 'W' "
    )
    query_to_table(query, "publications")


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


def create_query_tables():
    """
    Create tables for use in app queries

    TODO:
    max_priority_date = get_max_priority_date(min_patent_yrs)
    patent_life_filter = f"priority_date > {max_priority_date}"
    global_only_filter = "application_kind='W'"  # patent is global filing, aka "PCT"
    """
    logging.info("Creating patent tables for use in app queries")
    applications = (
        "SELECT "
        f"{','.join(FIELDS)} "
        f"FROM `{DATASET_ID}.publications` as pubs, "
        f"`{DATASET_ID}.gpr_publications` as gpr_pubs "
        "WHERE pubs.publication_number = gpr_pubs.publication_number "
    )
    entities = (
        "SELECT "
        "publication_number, "
        "ARRAY_AGG( "
        "struct(ocid, LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) as term, domain, confidence, source, character_offset_start) "
        "ORDER BY character_offset_start LIMIT 1 "
        ")[OFFSET(0)].* "
        f"FROM `{DATASET_ID}.gpr_annotations` a "
        f"LEFT JOIN `{DATASET_ID}.synonym_map` map ON a.preferred_name = map.synonym "
        "GROUP BY publication_number "
    )
    # query_to_table(applications, "applications")
    query_to_table(entities, "entities")


if __name__ == "__main__":
    # copy gpr_publications table
    copy_gpr_publications()

    # copy publications table
    copy_publications()

    # copy gpr_annotations table
    copy_gpr_annotations()  # depends on publications

    # create synonym_map table (for final entity names)
    create_synonym_map()

    # create the (small) tables against which the app will query
    create_query_tables()
