import json
import logging
import sys
from google.cloud import storage
from datetime import datetime, timedelta
import polars as pl
import psycopg2
from pydash import compact

import system

system.initialize()

from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID
from scripts.patents.initialize_patents import create_applications_table

storage_client = storage.Client()
db_client = BQDatabaseClient()

EXPORT_TABLES = {
    "biosym_annotations_source": None,
    "applications_tmp": "priority_date",
}

GCS_BUCKET = "biosym-patents"
# adjust this based on how large you want each shard to be
SHARD_SIZE = timedelta(days=730)


def export_bq_tables():
    """
    Export tables from BigQuery to GCS
    """
    logging.info("Exporting BigQuery tables to GCS")
    create_applications_table("applications_tmp")
    start_date = datetime(2000, 1, 1)
    end_date = datetime(2023, 1, 1)

    for table, date_column in EXPORT_TABLES.items():
        if date_column is None:
            destination_uri = f"gs://{GCS_BUCKET}/{table}.csv"
            db_client.export_table_to_storage(table, destination_uri)
        else:
            shared_table_name = f"{table}_shard_tmp"
            current_date = start_date
            while current_date < end_date:
                shard_end_date = current_date + SHARD_SIZE

                # Construct the SQL for exporting the shard
                shard_query = f"""
                    SELECT *
                    FROM `{BQ_DATASET_ID}.{table}`
                    WHERE {date_column} >= {int(current_date.strftime('%Y%m%d'))}
                    AND {date_column} < {int(shard_end_date.strftime('%Y%m%d'))}
                """
                db_client.select_to_table(shard_query, shared_table_name)

                # Define the destination in GCS
                destination_uri = f"gs://{GCS_BUCKET}/{table}_shard_{current_date.strftime('%Y%m%d')}.json"
                db_client.export_table_to_storage(shared_table_name, destination_uri)

                db_client.delete_table(shared_table_name)
                current_date = shard_end_date
            db_client.delete_table(shared_table_name)

    db_client.delete_table("applications_tmp")


def determine_data_type(value):
    if isinstance(value, int):
        return "INTEGER"
    elif isinstance(value, float):
        return "FLOAT"
    elif isinstance(value, bool):
        return "BOOLEAN"
    elif isinstance(value, list):
        if len(value) > 0:
            dt = determine_data_type(value[0])
            return f"{dt}[]"
        return "TEXT[]"
    else:  # default to TEXT for strings or other data types
        return "TEXT"


def import_into_psql():
    """
    Load data from a JSON file into a psql table
    """
    logging.info("Importing applications table (etc) into postgres")
    conn = psycopg2.connect(
        database="patents",
        host="localhost",
        port="5432",
    )
    cursor = conn.cursor()

    def create_table_if_not_exists(records, table_name: str):
        first_record = records[0]
        columns = []
        for key, value in first_record.items():
            pg_data_type = determine_data_type(value)
            columns.append(f'"{key}" {pg_data_type}')
        columns_str = ", ".join(columns)

        create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_str});"
        logging.info("Creating table: %s", create_table_query)
        cursor.execute(create_table_query)
        conn.commit()

    def transform(s):
        if isinstance(s, dict):
            return compact(s.values())[0]
        elif isinstance(s, list) or isinstance(s, pl.Series) and len(s) > 0:
            if isinstance(s[0], dict):
                return [compact(s1.values())[0] for s1 in s if len(s1) > 0]
        return s

    def load_data_from_json(file_blob: storage.Blob, table_name: str):
        lines = file_blob.download_as_text()
        records = [json.loads(line) for line in lines.split("\n") if line]
        df = pl.DataFrame(records)
        nono_columns = ["cited_by", "citation"]  # polars borks on these
        df = df.select(
            *[
                pl.col(c).apply(lambda s: transform(s))
                for c in df.columns
                if c not in nono_columns
            ]
        )
        create_table_if_not_exists(records, table_name)

        for record in df.to_dicts():
            columns = record.keys()
            values = [record[column] for column in columns]
            insert_statement = f"""
                INSERT INTO {table_name}({', '.join(['"' + c + '"' for c in columns])})
                VALUES ({', '.join(['%s'] * len(values))})
            """
            cursor.execute(insert_statement, tuple(values))

        conn.commit()

    bucket = storage_client.bucket(GCS_BUCKET)
    blobs: list[storage.Blob] = list(bucket.list_blobs())  # TODO: change to .json

    logging.info("Found %s blobs (%s)", len(blobs), bucket)

    for blob in blobs:
        table_name = [t for t in EXPORT_TABLES.keys() if t in str(blob.name)][0]
        logging.info("Adding data to table %s (%s)", table_name, blob.name)
        load_data_from_json(blob, table_name)

    # Close the cursor and connection
    cursor.close()
    conn.close()


def bq_to_psql():
    export_bq_tables()
    import_into_psql()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.bq_to_psql -export -import")
        sys.exit()

    if "-export" in sys.argv:
        export_bq_tables()

    if "-import" in sys.argv:
        import_into_psql()
