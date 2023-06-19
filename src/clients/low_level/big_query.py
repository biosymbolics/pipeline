from typing import Any, Callable
from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator
from google.api_core.exceptions import NotFound
import time
import logging
import os

logging.basicConfig(level=logging.INFO)

BQ_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
BQ_DATASET = "patents"
BQ_DATASET_ID = BQ_PROJECT + "." + BQ_DATASET


def execute_bg_query(query: str) -> RowIterator:
    """
    Execute BigQuery query

    Args:
        query (str): SQL query
    """
    client = bigquery.Client()
    logging.info("Starting query: %s", query)

    query_job = client.query(query)

    # Wait for the job to complete
    results = query_job.result()
    logging.info("Query complete")
    return results


def execute_with_retries(db_func: Callable[[], Any]):
    """
    Retry a function that interacts with BigQuery if it fails with a NotFound error

    Args:
        db_func (function): function that interacts with BigQuery
    """
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            db_func()
            break
        except NotFound as e:
            if retries < max_retries - 1:  # don't wait on last iteration
                time.sleep(1 * retries + 1)  # backoff
            retries += 1
        except Exception as e:
            raise e


def select_from_bg(query: str) -> list[dict]:
    """
    Execute a query and return the results as a list of dicts
    """
    results = execute_bg_query(query)
    rows = [dict(row) for row in results]

    logging.info("Rows returned: %s", len(rows))
    return rows


def query_to_bg_table(query: str, new_table_name: str):
    """
    Create a new table from a query

    Args:
        query (str): SQL query
        new_table_name (str): name of the new table
    """
    logging.info("Creating table %s", new_table_name)
    create_table_query = f"CREATE TABLE `{BQ_DATASET_ID}.{new_table_name}` AS {query};"
    execute_bg_query(create_table_query)
