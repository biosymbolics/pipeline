import json
import logging
import sys
from google.cloud import storage
from datetime import datetime, timedelta
import polars as pl
from pydash import compact
from clients.low_level.postgres import PsqlDatabaseClient

import system

system.initialize()

from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID

storage_client = storage.Client()
db_client = BQDatabaseClient()


APPLICATIONS_TABLE = "applications"
EXPORT_TABLES = {
    "biosym_annotations_source": None,
    APPLICATIONS_TABLE: "priority_date",  # shard by priority date
}

GCS_BUCKET = "biosym-patents"
# adjust this based on how large you want each shard to be
SHARD_SIZE = timedelta(days=730)


FIELDS = [
    # gpr_publications
    "gpr_pubs.publication_number as publication_number",
    "regexp_replace(gpr_pubs.publication_number, '-[^-]*$', '') as base_publication_number",  # for matching against approvals
    "abstract",
    "all_publication_numbers",
    "ARRAY(select regexp_replace(pn, '-[^-]*$', '') from UNNEST(all_publication_numbers) as pn) as all_base_publication_numbers",
    "application_number",
    "cited_by",
    "country",
    "embedding_v1 as embeddings",
    '"similar"',
    "title",
    "top_terms",
    "url",
    # publications
    "application_kind",
    # "assignee as assignee_raw",
    # "assignee_harmonized",
    "ARRAY(SELECT assignee.name FROM UNNEST(assignee_harmonized) as assignee) as assignees",
    "citation",
    "claims_localized as claims",
    "ARRAY(SELECT cpc.code FROM UNNEST(pubs.cpc) as cpc) as cpc_codes",
    "family_id",
    "filing_date",
    "grant_date",
    # "inventor as inventor_raw",
    # "inventor_harmonized",
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


def create_applications_table():
    """
    Create a table of patent applications in BigQuery
    (then exported and pulled into psql)
    """
    logging.info("Create a table of patent applications for use in app queries")

    client = BQDatabaseClient()
    client.delete_table(APPLICATIONS_TABLE)

    applications = f"""
        SELECT
        {','.join(FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` as pubs,
        `{BQ_DATASET_ID}.gpr_publications` as gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    client.create_from_select(applications, APPLICATIONS_TABLE)


def export_bq_tables():
    """
    Export tables from BigQuery to GCS
    """
    logging.info("Exporting BigQuery tables to GCS")
    create_applications_table()
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
                db_client.create_from_select(shard_query, shared_table_name)

                # Define the destination in GCS
                destination_uri = f"gs://{GCS_BUCKET}/{table}_shard_{current_date.strftime('%Y%m%d')}.json"
                db_client.export_table_to_storage(shared_table_name, destination_uri)

                db_client.delete_table(shared_table_name)
                current_date = shard_end_date


# alter table applications alter column priority_date type date USING priority_date::date;
TYPE_OVERRIDES = {
    "character_offset_start": "INTEGER",
    "character_offset_end": "INTEGER",
    "publication_date": "DATE",
    "filing_date": "DATE",
    "grant_date": "DATE",
    "priority_date": "DATE",
}


def determine_data_type(value, field: str | None = None):
    if field and field in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[field]

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
    client = PsqlDatabaseClient()

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

        first_record = records[0]
        columns = dict(
            [(f'"{k}"', determine_data_type(v, k)) for k, v in first_record.items()]
        )
        client.create_table(table_name, columns, exists_ok=True)
        client.insert_into_table(df.to_dicts(), table_name)

    bucket = storage_client.bucket(GCS_BUCKET)
    blobs: list[storage.Blob] = list(bucket.list_blobs())  # TODO: change to .json

    logging.info("Found %s blobs (%s)", len(blobs), bucket)

    for blob in blobs:
        matching_tables = [t for t in EXPORT_TABLES.keys() if t in str(blob.name)]
        table_name = matching_tables[0] if len(matching_tables) > 0 else None

        if not table_name:
            logging.info("Skipping blob %s", blob.name)
            continue

        logging.info("Adding data to table %s (%s)", table_name, blob.name)
        load_data_from_json(blob, table_name)


def copy_bq_to_psql():
    client = PsqlDatabaseClient()

    for table in EXPORT_TABLES.keys():
        client.truncate_table(table)

    export_bq_tables()
    import_into_psql()
    client.create_indices(
        [
            {
                "table": APPLICATIONS_TABLE,
                "column": "publication_number",
                "is_uniq": True,
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "abstract",
                "is_trgm": True,
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "title",
                "is_trgm": True,
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "priority_date",
            },
        ]
    )


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.bq_to_psql -export -import")
        sys.exit()

    if "-export" in sys.argv:
        export_bq_tables()

    if "-import" in sys.argv:
        import_into_psql()
