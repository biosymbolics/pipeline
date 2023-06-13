from google.cloud import bigquery
import logging
import os

logging.basicConfig(level=logging.INFO)

BQ_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
BQ_DATASET = "patents"
BQ_DATASET_ID = BQ_PROJECT + "." + BQ_DATASET


def execute_bg_query(query: str):
    """
    Execute BigQuery query

    Args:
        query (str): SQL query
    """
    # Create a client
    client = bigquery.Client()
    logging.info("Starting query: %s", query)

    query_job = client.query(query)

    # Wait for the job to complete
    results = query_job.result()
    logging.info("Query complete")
    return results


def select_from_bg(query: str) -> list[dict]:
    """
    Execute a query and return the results as a list of dicts
    """
    results = execute_bg_query(query)
    rows = [dict(row) for row in results]

    logging.info("Rows returned: %s", len(rows))
    return rows


def query_to_bg_table(query, new_table_name: str):
    """
    Create a new table from a query
    """
    logging.info("Creating table %s", new_table_name)
    create_table_query = f"CREATE TABLE `{BQ_DATASET_ID}.{new_table_name}` AS {query};"
    execute_bg_query(create_table_query)
