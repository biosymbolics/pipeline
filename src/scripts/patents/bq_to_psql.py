import json
import logging
import sys
from google.cloud import storage
from datetime import datetime, timedelta
import polars as pl
from pydash import compact
import logging

import system

system.initialize()

from clients.low_level.big_query import BQDatabaseClient, BQ_DATASET_ID
from clients.low_level.postgres import PsqlDatabaseClient

from ._constants import (
    APPLICATIONS_TABLE,
    GPR_ANNOTATIONS_TABLE,
    GPR_PUBLICATIONS_TABLE,
)

storage_client = storage.Client()
db_client = BQDatabaseClient()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPORT_TABLES = {
    # "biosym_annotations_source": None,
    APPLICATIONS_TABLE: {
        "column": "priority_date",
        "size": timedelta(days=730),
        "transform": lambda x: int(x.strftime("%Y%m%d")),
        "starting_value": datetime(2000, 1, 1),
        "ending_value": datetime(2023, 1, 1),
    },  # shard by priority date
    GPR_ANNOTATIONS_TABLE: {
        "column": "character_offset_start",
        "size": 500000,
        "starting_value": 0,
        "ending_value": 12592439,
        "transform": lambda x: x,
    },
}

GCS_BUCKET = "biosym-patents"
# adjust this based on how large you want each shard to be


INITIAL_COPY_FIELDS = [
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
    '"similar" as similar_patents',
    "title",
    "top_terms",
    "url",
    # publications
    "application_kind",
    "ARRAY(SELECT assignee.name FROM UNNEST(assignee_harmonized) as assignee) as assignees",
    "citation",
    "claims_localized as claims",
    "ARRAY(SELECT cpc.code FROM UNNEST(pubs.cpc) as cpc) as cpc_codes",
    "family_id",
    "ARRAY(SELECT inventor.name FROM UNNEST(inventor_harmonized) as inventor) as inventors",
    "ARRAY(SELECT ipc.code FROM UNNEST(ipc) as ipc) as ipc_codes",
    "kind_code",
    "pct_number",
    "priority_claim",
    "priority_date",
    "spif_application_number",
    "spif_publication_number",
]


def create_patent_applications_table():
    """
    Create a table of patent applications in BigQuery
    (then exported and pulled into psql)
    """
    logging.info("Create a table of patent applications for use in app queries")

    client = BQDatabaseClient()
    client.delete_table(APPLICATIONS_TABLE)

    applications = f"""
        SELECT {','.join(INITIAL_COPY_FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` pubs,
        `{BQ_DATASET_ID}.{GPR_PUBLICATIONS_TABLE}` gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    client.create_from_select(applications, APPLICATIONS_TABLE)


def export_bq_tables(today):
    """
    Export tables from BigQuery to GCS
    """
    logging.info("Exporting BigQuery tables to GCS")
    create_patent_applications_table()

    for table, shard_spec in EXPORT_TABLES.items():
        if shard_spec is None:
            destination_uri = f"gs://{GCS_BUCKET}/{table}.csv"
            db_client.export_table_to_storage(table, destination_uri)
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
                db_client.create_from_select(shard_query, shared_table_name)

                # Define the destination in GCS
                destination_uri = (
                    f"gs://{GCS_BUCKET}/{today}/{table}_shard_{current_shard}.json"
                )
                db_client.export_table_to_storage(shared_table_name, destination_uri)

                db_client.delete_table(shared_table_name)
                current_shard = shard_end


# alter table applications alter column priority_date type date USING priority_date::date;
TYPE_OVERRIDES = {
    "character_offset_start": "INTEGER",
    "character_offset_end": "INTEGER",
    # "publication_date": "DATE",
    # "filing_date": "DATE",
    # "grant_date": "DATE",
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


def import_into_psql(today: str):
    """
    Load data from a JSON file into a psql table
    """
    logging.info("Importing applications table (etc) into postgres")
    client = PsqlDatabaseClient()

    # truncate table copy in postgres
    for table in EXPORT_TABLES.keys():
        client.truncate_table(table)

    def transform(s, c: str):
        if c.endswith("_date") and s == 0:
            logger.info("RETURNING NONE for %s %s", c, s)
            return None
        if isinstance(s, dict):
            # TODO: this is a hack, only works because the first value is currently always the one we want
            return compact(s.values())[0]
        elif isinstance(s, list) or isinstance(s, pl.Series) and len(s) > 0:
            if isinstance(s[0], dict):
                return [compact(s1.values())[0] for s1 in s if len(s1) > 0]
        return s

    def load_data_from_json(file_blob: storage.Blob, table_name: str):
        lines = file_blob.download_as_text()
        records = [json.loads(line) for line in lines.split("\n") if line]
        df = pl.DataFrame(records)
        nono_columns = [
            "cited_by",
            "citation",
            "publication_date",
        ]  # requires work on transformation
        df = df.select(
            *[
                pl.col(c).apply(lambda s: transform(s, c))
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

    bucket = storage_client.bucket(GCS_BUCKET)  # {GCS_BUCKET}/{today}
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


def copy_bq_to_psql():
    today = datetime.now().strftime("%Y%m%d")
    export_bq_tables(today)
    import_into_psql(today)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.bq_to_psql -export -import")
        sys.exit()

    today = datetime.now().strftime("%Y%m%d")

    if "-export" in sys.argv:
        export_bq_tables(today)

    if "-import" in sys.argv:
        import_into_psql(today)
