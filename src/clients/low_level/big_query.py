"""
Low-level BigQuery client
"""
from typing import Any, Callable
from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator
from google.api_core.exceptions import NotFound
import time
import logging
import os
import polars as pl

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

    Args:
        query (str): SQL query
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


def get_table(table_name: str) -> bigquery.Table:
    """
    Check if a table exists
    """
    logging.info("Grabbling table %s", table_name)
    client = bigquery.Client()
    table = client.get_table(f"{BQ_DATASET_ID}.{table_name}")
    return table


def select_insert_into_bg_table(select_query: str, table_name: str):
    """
    Insert rows into a table from a select query

    Args:
        select_query (str): select query
        table_name (str): name of the table
    """
    query = f"""
        INSERT INTO `f"{BQ_DATASET_ID}.{table_name}"`
        {select_query}
    """
    execute_bg_query(query)


def insert_into_bg_table(df: pl.DataFrame, table_name: str):
    """
    Insert rows into a table from a dataframe
    - validate that the table exists
    - insert the df rows into the table

    Args:
        df (pl.DataFrame): dataframe to insert
        table_name (str): name of the table
    """
    logging.info("Inserting into table %s", table_name)
    table = get_table(table_name)

    # insert the df rows into the table
    client = bigquery.Client()
    client.insert_rows_from_dataframe(table, df.to_pandas())


def upsert_into_bg_table(
    df: pl.DataFrame,
    table_name: str,
    id_fields: list[str],
    insert_fields: list[str],
    on_conflict: str,
):
    """
    Upsert rows into a table from a dataframe
    - validate that the table exists
    - upsert the df rows into the table

    Args:
        df (pl.DataFrame): dataframe to upsert
        table_name (str): name of the table
    """
    logging.info("Upserting into table %s", table_name)

    # Define a temporary table name
    tmp_table_name = table_name + "_tmp"

    client = bigquery.Client()

    # Create a temporary table
    pd_df = df.to_pandas(use_pyarrow_extension_array=True)
    client.load_table_from_dataframe(
        pd_df, f"{BQ_DATASET_ID}.{tmp_table_name}"
    ).result()

    # Identity JOIN
    identity_join = " AND ".join(
        [f"target.{field} = source.{field}" for field in id_fields]
    )

    # insert if no conflict
    insert = f"""
        INSERT ({', '.join(insert_fields)})
        VALUES ({', '.join([f'source.{field}' for field in insert_fields])})
    """

    # Use a MERGE statement to perform the upsert operation
    sql = f"""
    MERGE {BQ_DATASET_ID}.{table_name} AS target
    USING {BQ_DATASET_ID}.{tmp_table_name} AS source
    ON {identity_join}
    WHEN MATCHED THEN
        {on_conflict}
    WHEN NOT MATCHED THEN {insert}
    """

    logging.info("Running query: %s", sql)
    execute_bg_query(sql)

    # Delete the temporary table
    delete_bg_table(tmp_table_name)


def delete_bg_table(table_name: str):
    """
    Delete a table (if exists)

    Args:
        table_name (str): name of the table
    """
    logging.info("Deleting table %s", table_name)
    delete_table_query = f"DROP TABLE IF EXISTS `{BQ_DATASET_ID}.{table_name}`;"
    execute_bg_query(delete_table_query)
