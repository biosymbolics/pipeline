import asyncio
import json
import logging
import sys
from typing import Any, Sequence, TypeGuard
from datetime import datetime
import polars as pl
from pydash import compact
import logging
import google.cloud.storage as storage

import system

system.initialize()

from constants.umls import NER_ENTITY_TYPES
from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import PUBLICATION_NUMBER_MAP_TABLE, SOURCE_BIOSYM_ANNOTATIONS_TABLE

from .constants import (
    GPR_ANNOTATIONS_TABLE,
    GPR_PUBLICATIONS_TABLE,
)
from .utils import determine_data_type


storage_client = storage.Client()
db_client = BQDatabaseClient()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

APPLICATIONS_TABLE = "applications"


def is_dict_list(obj: Any) -> TypeGuard[list[dict[str, Any]]]:
    """
    Checks if an object is a list of dicts

    Args:
        obj (Any): object to check

    Returns:
        bool: True if the object is a list of dicts
    """
    return isinstance(obj, list) and all(isinstance(x, dict) for x in obj)


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
    SOURCE_BIOSYM_ANNOTATIONS_TABLE: {
        "column": "domain",
        "values": NER_ENTITY_TYPES,
    },
    # "applications": {
    #     "column": "priority_date",
    #     "size": timedelta(days=730),
    #     "transform": lambda x: int(x.strftime("%Y%m%d")),
    #     "starting_value": datetime(2000, 1, 1),
    #     "ending_value": datetime(2023, 1, 1),
    # },
    # GPR_ANNOTATIONS_TABLE: {
    #     "column": "confidence",
    #     "size": 0.0125,
    #     "starting_value": 0.774,
    #     "ending_value": 0.91,  # max 0.90
    #     "transform": lambda x: x,
    # },
}

GCS_BUCKET = "biosym-patents"
# adjust this based on how large you want each shard to be


async def create_patent_applications_table():
    """
    Create a table of patent applications in BigQuery
    (which is then exported and imported into psql)
    """
    logging.info("Create a table of patent applications on BigQuery for sharding")

    client = BQDatabaseClient()
    await client.delete_table(APPLICATIONS_TABLE)

    # NOTE: as of 01/08, left join has not been tested (might cause errors down the line)
    applications = f"""
        SELECT {','.join(PATENT_APPLICATION_FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` pubs
        LEFT JOIN `{BQ_DATASET_ID}.{GPR_PUBLICATIONS_TABLE}` gpr_pubs on pubs.publication_number = gpr_pubs.publication_number
    """
    await client.create_from_select(applications, APPLICATIONS_TABLE)


async def shared_and_export(
    shard_query: str, shared_table_name: str, table: str, value: str
):
    """
    Create a shared table, export to GCS and delete
    """
    await db_client.create_from_select(shard_query, shared_table_name)
    destination_uri = f"gs://{GCS_BUCKET}/{today}/{table}_shard_{value}.json"
    db_client.export_table_to_storage(shared_table_name, destination_uri)
    await db_client.delete_table(shared_table_name)


async def export_bq_tables():
    """
    Export tables from BigQuery to GCS
    """
    logging.info("Exporting BigQuery tables to GCS")
    await create_patent_applications_table()

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
                await shared_and_export(shard_query, shared_table_name, table, value)
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
                await shared_and_export(
                    shard_query, shared_table_name, table, current_shard
                )


async def import_into_psql(today: str):
    """
    Load data from a JSON file into a psql table
    """
    logging.info("Importing applications table (etc) into postgres")
    client = PsqlDatabaseClient()

    # truncate table copy in postgres
    for table in EXPORT_TABLES.keys():
        await client.truncate_table(table)

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

    async def load_data_from_json(file_blob: storage.Blob, table_name: str):
        lines = file_blob.download_as_text()
        records = [json.loads(line) for line in lines.split("\n") if line]
        df = pl.DataFrame(records)
        df = df.select(
            *[pl.col(c).apply(lambda s: transform(s, c)) for c in df.columns]
        )

        first_record = records[0]
        columns = dict(
            [(f'"{k}"', determine_data_type(v, k)) for k, v in first_record.items()]
        )
        await client.create_table(table_name, columns, exists_ok=True)
        await client.insert_into_table(df.to_dicts(), table_name)

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
        await load_data_from_json(blob, table_name)


async def create_derived_tables(client: PsqlDatabaseClient):
    """
    Create derived tables
    """
    # create mapping between the main publication number (WPO) and all other publication numbers
    await client.create_from_select(
        f"select publication_number, unnest(all_base_publication_numbers) as other_publication_number from {APPLICATIONS_TABLE}",
        PUBLICATION_NUMBER_MAP_TABLE,
    )
    await client.create_indices(
        [
            {
                "table": PUBLICATION_NUMBER_MAP_TABLE,
                "column": "publication_number",
            },
            {
                "table": PUBLICATION_NUMBER_MAP_TABLE,
                "column": "other_publication_number",
            },
        ]
    )


async def copy_bq_to_psql():
    """
    Copy data from BigQuery to psql + create index
    """
    today = datetime.now().strftime("%Y%m%d")
    await export_bq_tables()
    await import_into_psql(today)

    client = PsqlDatabaseClient()
    await create_derived_tables(client)

    await client.create_indices(
        [
            {
                "table": APPLICATIONS_TABLE,
                "column": "publication_number",
                "is_uniq": True,
            },
            {
                "table": GPR_ANNOTATIONS_TABLE,
                "column": "preferred_term",
                "is_lower": True,
            },
            {
                "table": GPR_ANNOTATIONS_TABLE,
                "column": "domain",
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
        asyncio.run(export_bq_tables())

    if "-import" in sys.argv:
        asyncio.run(import_into_psql(today))
