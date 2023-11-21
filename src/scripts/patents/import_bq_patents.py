import json
import logging
import sys
from typing import Sequence
from google.cloud import storage
from datetime import datetime, timedelta
import polars as pl
from pydash import compact
import logging

import system

system.initialize()

from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import APPLICATIONS_TABLE
from typings.core import is_dict_list

from .constants import (
    GPR_ANNOTATIONS_TABLE,
    GPR_PUBLICATIONS_TABLE,
)
from .utils import determine_bq_data_type


storage_client = storage.Client()
db_client = BQDatabaseClient()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PATENT_APPLICATION_FIELDS = [
    # ** FROM gpr_publications **
    "gpr_pubs.publication_number as publication_number",
    "regexp_replace(gpr_pubs.publication_number, '-[^-]*$', '') as base_publication_number",  # for matching against approvals
    "abstract",
    "all_publication_numbers",  # all the country-specific patents
    "ARRAY(select regexp_replace(pn, '-[^-]*$', '') from UNNEST(all_publication_numbers) as pn) as all_base_publication_numbers",
    "application_number",
    "country",
    "ARRAY(SELECT s.publication_number FROM UNNEST(similar) as s) as similar_patents",
    "title",
    "url",
    # **FROM PUBLICATIONS **
    "ARRAY(SELECT assignee.name FROM UNNEST(assignee_harmonized) as assignee) as assignees",
    "claims_localized as claims",
    "family_id",
    "ARRAY(SELECT inventor.name FROM UNNEST(inventor_harmonized) as inventor) as inventors",
    "ARRAY(SELECT ipc.code FROM UNNEST(ipc) as ipc) as ipc_codes",
    "priority_date",
]

EXPORT_TABLES = {
    "biosym_annotations_source": {
        "column": "domain",
        "values": ["attributes", "compounds", "diseases", "mechanisms"],
    },
    APPLICATIONS_TABLE: {
        "column": "priority_date",
        "size": timedelta(days=730),
        "transform": lambda x: int(x.strftime("%Y%m%d")),
        "starting_value": datetime(2000, 1, 1),
        "ending_value": datetime(2023, 1, 1),
    },
    GPR_ANNOTATIONS_TABLE: {
        "column": "confidence",
        "size": 0.0125,
        "starting_value": 0.774,
        "ending_value": 0.91,  # max 0.90
        "transform": lambda x: x,
    },
}

GCS_BUCKET = "biosym-patents"
# adjust this based on how large you want each shard to be


def create_patent_applications_table():
    """
    Create a table of patent applications in BigQuery
    (which is then exported and imported into psql)
    """
    logging.info("Create a table of patent applications on BigQuery for sharding")

    client = BQDatabaseClient()
    client.delete_table(APPLICATIONS_TABLE)

    applications = f"""
        SELECT {','.join(PATENT_APPLICATION_FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` pubs,
        `{BQ_DATASET_ID}.{GPR_PUBLICATIONS_TABLE}` gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    client.create_from_select(applications, APPLICATIONS_TABLE)


def shared_and_export(shard_query: str, shared_table_name: str, table: str, value: str):
    """
    Create a shared table, export to GCS and delete
    """
    db_client.create_from_select(shard_query, shared_table_name)
    destination_uri = f"gs://{GCS_BUCKET}/{today}/{table}_shard_{value}.json"
    db_client.export_table_to_storage(shared_table_name, destination_uri)
    db_client.delete_table(shared_table_name)


def export_bq_tables():
    """
    Export tables from BigQuery to GCS
    """
    logging.info("Exporting BigQuery tables to GCS")
    create_patent_applications_table()

    for table, shard_spec in EXPORT_TABLES.items():
        if shard_spec is None:
            destination_uri = f"gs://{GCS_BUCKET}/{table}.csv"
            db_client.export_table_to_storage(table, destination_uri)
        if "values" in shard_spec:
            for value in shard_spec["values"]:
                shared_table_name = f"{table}_shard_tmp"
                shard_query = f"""
                    SELECT * FROM `{BQ_DATASET_ID}.{table}`
                    WHERE {shard_spec["column"]} = '{value}'
                """
                shared_and_export(shard_query, shared_table_name, table, value)
        else:
            shared_table_name = f"{table}_shard_tmp"
            current_shard = shard_spec["starting_value"]
            while current_shard < shard_spec["ending_value"]:
                shard_end = current_shard + shard_spec["size"]

                # Construct the SQL for exporting the shard
                shard_query = f"""
                    SELECT *
                    FROM `{BQ_DATASET_ID}.{table}`
                    WHERE {shard_spec["column"]} >= {shard_spec["transform"](current_shard)}
                    AND {shard_spec["column"]} < {shard_spec["transform"](shard_end)}
                """
                shared_and_export(shard_query, shared_table_name, table, current_shard)


def import_into_psql(today: str):
    """
    Load data from a JSON file into a psql table
    """
    logging.info("Importing applications table (etc) into postgres")
    client = PsqlDatabaseClient()

    # truncate table copy in postgres
    for table in EXPORT_TABLES.keys():
        client.truncate_table(table)

    def transform(value: str | dict | int | float | Sequence | pl.Series, col: str):
        if col.endswith("_date") and isinstance(value, int) and value == 0:
            # return none for zero-valued dates
            return None
        if isinstance(value, dict):
            # TODO: this is a hack, only works because the first value is currently always the one we want
            return compact(value.values())[0]
        elif (isinstance(value, list) or isinstance(value, pl.Series)) and len(
            value
        ) > 0:
            if is_dict_list(value):
                return [compact(v.values())[0] for v in value if len(value) > 0]
        return value

    def load_data_from_json(file_blob: storage.Blob, table_name: str):
        lines = file_blob.download_as_text()
        records = [json.loads(line) for line in lines.split("\n") if line]
        df = pl.DataFrame(records)
        df = df.select(
            *[pl.col(c).map_elements(lambda s: transform(s, c)) for c in df.columns]
        )

        first_record = records[0]
        columns = dict(
            [(f'"{k}"', determine_bq_data_type(v, k)) for k, v in first_record.items()]
        )
        client.create_table(table_name, columns, exists_ok=True)
        client.insert_into_table(df.to_dicts(), table_name)

    bucket = storage_client.bucket(GCS_BUCKET)
    blobs: list[storage.Blob] = list(bucket.list_blobs(prefix=today))

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
    """
    Copy data from BigQuery to psql + create index
    """
    today = datetime.now().strftime("%Y%m%d")
    export_bq_tables()
    import_into_psql(today)

    client = PsqlDatabaseClient()
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
                "is_tgrm": True,
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "title",
                "is_tgrm": True,
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "priority_date",
            },
            {
                "table": APPLICATIONS_TABLE,
                "column": "all_base_publication_numbers",
                "is_gin": True,
            },
        ]
    )

    client.create_indices(
        [
            {
                "table": GPR_ANNOTATIONS_TABLE,
                "column": "preferred_term",
                "is_lower": True,
            },
            {"table": GPR_ANNOTATIONS_TABLE, "column": "publication_number"},
        ]
    )


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.bq_to_psql -export -import")
        sys.exit()

    today = datetime.now().strftime("%Y%m%d")

    if "-export" in sys.argv:
        export_bq_tables()

    if "-import" in sys.argv:
        import_into_psql(today)
