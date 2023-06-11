from google.cloud import bigquery
import logging

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


def copy_from_query(query, new_table_name: str):
    """
    Create a new table from a query
    """
    create_table_query = f"CREATE TABLE `{DATASET_ID}.{new_table_name}` AS {query};"
    __run_bq_query(create_table_query)


def copy_gpr_publications():
    query = (
        "SELECT * FROM `patents-public-data.google_patents_research.publications` "
        "WHERE EXISTS "
        f'(SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))'
    )
    copy_from_query(query, "gpr_publications")


def copy_gpr_annotations():
    COMMON_NAMES = (
        "water",
        "drug",
        "drugs",
        "mixture",
        "acid",
        "compound",
        "method",
        "process",
        "fluid",
        "mixing",
        "solution",
        "material",
        "hydrogen",
        "solid",
        "substance",
        "cells",
        "disease",
        "treatment",
        "gel",
        "product",
        "therapeutic",
        "liquid",
        "media",
        "medium",
        "culture",
        "protein",
        "cell",
        "oxygen",
        "carbon",
        "sodium",
        "solid",
        "sodium chloride",
        "sodium",
        "solvent",
        "sulfur",
        "preservative agent",
        "adhesive" "base",
        "pharmaceutical composition",
        "pharmaceutical",
        "pharmaceutical preparation",
        "sample",
        "starch",
        "solvent",
        "marker",
    )
    SUPPRESSED_DOMAINS = (
        "inorgmat",
        "nutrients",
    )
    query = (
        "SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations "
        f"JOIN `{DATASET_ID}.gpr_publications` AS local_publications "
        "ON local_publications.publication_number = annotations.publication_number "
        "WHERE annotations.confidence > 0.5 "
        f"AND preferred_name not in {COMMON_NAMES} "
        f"AND domain not in {SUPPRESSED_DOMAINS} "
    )
    copy_from_query(query, "gpr_annotations")


def copy_publications():

    query = (
        "SELECT publications.* FROM `patents-public-data.patents.publications` as publications "
        f"JOIN `{DATASET_ID}.gpr_publications` AS local_gpr "
        "ON local_gpr.publication_number = publications.publication_number "
        "WHERE application_kind = 'W' "
    )
    copy_from_query(query, "publications")


if __name__ == "__main__":
    # copy_gpr_publications()
    # copy_publications()
    copy_gpr_annotations()  # depends on publications
