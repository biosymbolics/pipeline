"""
Low-level BigQuery client
"""
from typing import Mapping, TypeVar
from google.cloud import bigquery
from google.cloud.bigquery import job
from google.api_core.exceptions import NotFound
from google.oauth2.service_account import Credentials
import time
import logging
import os
import polars as pl
import json

from clients.low_level.boto3 import get_boto_client
from typings.core import is_string_list

from clients.low_level.database import DatabaseClient, ExecuteResult
from utils.classes import overrides, nonoverride

T = TypeVar("T", bound=Mapping)

BQ_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
BQ_DATASET = "patents"
BQ_DATASET_ID = BQ_PROJECT + "." + BQ_DATASET

CREDENTIALS_PATH = "/biosymbolics/pipeline/google/credentials"
DEFAULT_USE_SERVICE_ACCOUNT = False

logger = logging.getLogger(__name__)


class BigQueryClient(bigquery.Client):
    def __init__(self, use_service_account: bool):
        if use_service_account:
            creds = self.get_google_credentials_from_ssm(CREDENTIALS_PATH)
            super().__init__(credentials=creds)
        else:
            super().__init__()

    @classmethod
    def get_google_credentials_from_ssm(cls, parameter_name: str):
        ssm_client = get_boto_client("ssm")
        response = ssm_client.get_parameter(Name=parameter_name, WithDecryption=True)
        value = response["Parameter"]["Value"]
        cred_info = json.loads(value)
        creds = Credentials.from_service_account_info(cred_info)
        return creds


class BQDatabaseClient(DatabaseClient):
    def __init__(self, use_service_account: bool = False):
        self.client = BigQueryClient(use_service_account=use_service_account)

    @staticmethod
    @overrides(DatabaseClient)
    def get_table_id(table_name: str) -> str:
        """
        Get the full table name (including project and dataset) from a table name
        """
        return (
            f"{BQ_DATASET_ID}.{table_name}"
            if BQ_DATASET_ID not in table_name
            else table_name
        )

    @overrides(DatabaseClient)
    def is_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists
        """
        try:
            self.get_table(table_name)
            return True
        except NotFound:
            return False

    @overrides(DatabaseClient)
    def execute_query(self, query: str) -> ExecuteResult:
        """
        Execute BigQuery query

        Args:
            query (str): SQL query
        """
        logging.info("Starting query: %s", query)

        query_job = self.client.query(query)

        # Wait for the job to complete
        results = list(query_job.result())
        logging.info("Query complete")
        return {"data": results, "columns": []}

    @overrides(DatabaseClient)
    def _insert(self, table_name: str, records: list[T]):
        self.client.insert_rows(self.get_table(table_name), records)

    @overrides(DatabaseClient)
    def _create(
        self, table_name: str, columns: list[str] | dict[str, str]
    ) -> bigquery.Table:
        """
        Simple create table function, makes up schema based on column names

        Args:
            table_name (str): name of the table
            columns (list[str] | dict): list of columns or schema
        """
        table_id = self.get_table_id(table_name)
        if is_string_list(columns):
            columns = columns
            schema = [
                bigquery.SchemaField(
                    field_name, "DATE" if field_name.endswith("_date") else "STRING"
                )
                for field_name in columns
            ]
        elif isinstance(columns, dict):
            schema = [bigquery.SchemaField(c, t) for c, t in columns.items()]
        else:
            raise Exception("Invalid columns")

        new_table = bigquery.Table(table_id, schema)
        return self.client.create_table(new_table)

    @nonoverride
    def get_table(self, table_name: str) -> bigquery.Table:
        """
        Check if a table exists,
        throws exception if it doesn't
        """
        table_id = self.get_table_id(table_name)
        logging.info("Grabbing table %s", table_id)
        table = self.client.get_table(table_id)
        return table

    @nonoverride
    def insert_df_into_table(self, df: pl.DataFrame, table_name: str):
        """
        Insert rows into a table from a dataframe
        - validate that the table exists
        - insert the df rows into the table

        Args:
            df (pl.DataFrame): dataframe to insert
            table_name (str): name of the table
        """
        table_id = self.get_table_id(table_name)
        logging.info("Inserting into table %s", table_id)

        # insert the df rows into the table
        self.client.insert_rows_from_dataframe(table_id, df.to_pandas())

    @nonoverride
    def upsert_df_into_table(
        self,
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
        table_id = self.get_table_id(table_name)
        logging.info("Upserting into table %s", table_name)

        # Define a temporary table name
        tmp_table_id = table_id + "_tmp"

        # truncate if exists to avoid dups
        self.truncate_table(tmp_table_id)

        # Create and populate temp table
        pd_df = df.to_pandas(use_pyarrow_extension_array=True)
        self.client.load_table_from_dataframe(pd_df, tmp_table_id).result()

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
        try:
            self.execute_query(sql)
        except Exception as e:
            logging.error("Error upserting rows: %s", e)
            self.delete_table(tmp_table_id)
            raise e
        finally:
            # Delete the temporary table
            self.delete_table(tmp_table_id)

    @nonoverride
    def export_table_to_storage(self, table_name: str, destination_uri: str):
        """
        Export a table to storage (GCS for now, as CSV)

        Args:
            table_name (str): name of the table
            destination_uri (str): storage destination URI
        """
        job_config = job.ExtractJobConfig()
        job_config.destination_format = (
            bigquery.DestinationFormat.NEWLINE_DELIMITED_JSON
        )
        extract_job = self.client.extract_table(
            self.get_table(table_name), destination_uri, job_config=job_config
        )
        extract_job.result()  # Wait for the job to complete
