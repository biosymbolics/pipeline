"""
Low-level BigQuery client
"""
from typing import Any, Callable, Mapping, TypeVar
from google.cloud import bigquery
from google.cloud.bigquery.table import RowIterator
from google.api_core.exceptions import NotFound
from google.oauth2.service_account import Credentials
import time
import logging
import os
import polars as pl
import json

from clients.low_level.boto3 import get_boto_client
from utils.list import batch
from typings.core import is_string_list

BQ_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
BQ_DATASET = "patents"
BQ_DATASET_ID = BQ_PROJECT + "." + BQ_DATASET

CREDENTIALS_PATH = "/biosymbolics/pipeline/google/credentials"

logger = logging.getLogger(__name__)


class BigQueryClient(bigquery.Client):
    def __init__(self):
        creds = self.get_google_credentials_from_ssm(CREDENTIALS_PATH)
        super().__init__(credentials=creds)

    @classmethod
    def get_google_credentials_from_ssm(cls, parameter_name: str):
        ssm_client = get_boto_client("ssm")
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        value = response["Parameter"]["Value"]
        cred_info = json.loads(value)
        creds = Credentials.from_service_account_info(cred_info)
        return creds


def get_table_id(table_name: str) -> str:
    """
    Get the full table name (including project and dataset) from a table name
    """
    return (
        f"{BQ_DATASET_ID}.{table_name}"
        if BQ_DATASET_ID not in table_name
        else table_name
    )


def execute_bg_query(query: str) -> RowIterator:
    """
    Execute BigQuery query

    Args:
        query (str): SQL query
    """
    client = BigQueryClient()
    logging.info("Starting query: %s", query)

    query_job = client.query(query)

    # Wait for the job to complete
    results = query_job.result()
    logging.info("Query complete")
    return results


def select_from_bg(query: str) -> list[dict]:
    """
    Execute a query and return the results as a list of dicts
    (must include provide fully qualified table name in query)

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
    new_table_id = get_table_id(new_table_name)
    create_table_query = f"CREATE or REPLACE TABLE `{new_table_id}` AS {query};"
    execute_bg_query(create_table_query)


def get_bg_table(table_name: str) -> bigquery.Table:
    """
    Check if a table exists,
    throws exception if it doesn't
    """
    table_id = get_table_id(table_name)
    logging.info("Grabbing table %s", table_id)
    client = BigQueryClient()
    table = client.get_table(table_id)
    return table


def select_insert_into_bg_table(select_query: str, table_name: str):
    """
    Insert rows into a table from a select query

    Args:
        select_query (str): select query
        table_name (str): name of the table
    """
    table_id = get_table_id(table_name)
    query = f"INSERT INTO `{table_id}` {select_query}"
    logging.info("Inserting via query (%s) into table %s", query, table_id)
    execute_bg_query(query)


def insert_df_into_bg_table(df: pl.DataFrame, table_name: str):
    """
    Insert rows into a table from a dataframe
    - validate that the table exists
    - insert the df rows into the table

    Args:
        df (pl.DataFrame): dataframe to insert
        table_name (str): name of the table
    """
    table_id = get_table_id(table_name)
    logging.info("Inserting into table %s", table_id)

    # insert the df rows into the table
    client = bigquery.Client()
    client.insert_rows_from_dataframe(table_id, df.to_pandas())


T = TypeVar("T", bound=Mapping)


def insert_into_bg_table(records: list[T], table_name: str, batch_size: int = 1000):
    """
    Insert rows into a table from a list of records

    Args:
        records (list[dict]): list of records to insert
        table_name (str): name of the table
        batch_size (int, optional): number of records to insert per batch. Defaults to 1000.
    """

    def __insert_into_bg_table(records: list[T], table: bigquery.Table):
        """
        Insert records into a table
        """
        client = bigquery.Client()

        try:
            errors = client.insert_rows(table, records)
            if errors:
                raise Exception("Error inserting rows")
        except Exception as e:
            logging.error("Error inserting rows: %s", e)
            raise e

        logging.info("Successfully inserted %s rows", len(records))

    batched = batch(records, batch_size)
    table = get_bg_table(table_name)

    for i, b in enumerate(batched):
        logging.info("Inserting batch %s into table %s", i, table_name)
        __insert_into_bg_table(b, table)


def upsert_df_into_bg_table(
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
        id_fields (list[str]): list of fields to use for the identity join
        insert_fields (list[str]): list of fields to insert
        on_conflict (str): conflict resolution strategy
    """
    table_id = get_table_id(table_name)
    logging.info("Upserting into table %s", table_name)

    # Define a temporary table name
    tmp_table_id = table_id + "_tmp"

    client = bigquery.Client()

    # Create a temporary table
    pd_df = df.to_pandas(use_pyarrow_extension_array=True)
    client.load_table_from_dataframe(pd_df, tmp_table_id).result()

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
    MERGE {table_id} AS target
    USING {tmp_table_id} AS source
    ON {identity_join}
    WHEN MATCHED THEN
        {on_conflict}
    WHEN NOT MATCHED THEN {insert}
    """

    logging.info("Running query: %s", sql)
    execute_bg_query(sql)

    # Delete the temporary table
    delete_bg_table(tmp_table_id)


def delete_bg_table(table_name: str):
    """
    Delete a table (if exists)

    Args:
        table_name (str): name of the table
    """
    table_id = get_table_id(table_name)
    logging.info("Deleting table %s", table_name)
    delete_table_query = f"DROP TABLE IF EXISTS `{table_id}`;"
    execute_bg_query(delete_table_query)


def truncate_bg_table(table_name: str):
    """
    Truncate a table (if exists)

    Args:
        table_name (str): name of the table
    """
    exists = does_table_exist(table_name)
    if not exists:
        logging.warning("Table %s does not exist and thus not truncating", table_name)
        return

    table_id = get_table_id(table_name)
    logging.info("Truncating table %s", table_id)
    truncate_table_query = f"TRUNCATE TABLE `{table_id}`;"
    execute_bg_query(truncate_table_query)


def __create_table(
    table_name: str, schema_or_cols: list[str] | list[bigquery.SchemaField]
) -> bigquery.Table:
    """
    Simple create table function, makes up schema based on column names

    Args:
        table_name (str): name of the table
        schema_or_cols (list[str] | list[bigquery.SchemaField]): list of columns or schema
    """
    table_id = get_table_id(table_name)
    client = bigquery.Client()

    if is_string_list(schema_or_cols):
        columns = schema_or_cols
        schema = [
            bigquery.SchemaField(
                field_name, "DATE" if field_name.endswith("_date") else "STRING"
            )
            for field_name in columns
        ]
    else:
        schema = schema_or_cols

    new_table = bigquery.Table(table_id, schema)
    return client.create_table(new_table)


##### Higher level functions #####


def does_table_exist(table_name: str) -> bool:
    """
    Check if a table exists
    """
    try:
        get_bg_table(table_name)
        return True
    except NotFound:
        return False


def create_bq_table(
    table_name: str,
    schema: list[str] | list[bigquery.SchemaField],
    exists_ok: bool = True,
    truncate_if_exists: bool = False,
) -> bigquery.Table:
    """
    Create a BigQuery table

    Args:
        table_name (str): name of the table (with or without dataset prefix)
        schema (list[str] | list[bigquery.SchemaField]): list of column names or list of SchemaField objects
        exists_ok (bool): if True, do not raise an error if the table already exists
        truncate_if_exists (bool): if True, truncate the table if it already exists
    """
    if truncate_if_exists and not exists_ok:
        raise Exception("Cannot truncate if exists if exists_ok is False")

    if does_table_exist(table_name):
        if truncate_if_exists:
            truncate_bg_table(table_name)
        elif not exists_ok:
            raise Exception(f"Table {table_name} already exists")
        else:
            logging.info("Table %s already exists", table_name)

        return get_bg_table(table_name)

    return __create_table(table_name, schema)


##### Utils #####


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
