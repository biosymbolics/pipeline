import os
import sys
from typing import Collection
import csv
import dedupe

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient

COMPANY_DEDUP_FILE = "company_dedup.csv"


def dedup_interactive(
    data_d: dict,
    variables: Collection,
    output_file: str = COMPANY_DEDUP_FILE,
    training_file="company_dedup_training.json",
    settings_file="company_dedup_settings",
):
    deduper = dedupe.Dedupe(variables)

    if os.path.isfile(training_file):
        with open(training_file) as tf:
            deduper.prepare_training(data_d, tf)
    else:
        deduper.prepare_training(data_d)
    dedupe.console_label(deduper)
    deduper.train()

    with open(training_file, "w") as tf:
        deduper.write_training(tf)

    with open(settings_file, "wb") as sf:
        deduper.write_settings(sf)

    clustered_dupes = deduper.partition(data_d, 0.5)
    cluster_membership = {}
    for cluster_id, (records, scores) in enumerate(clustered_dupes):
        for record_id, score in zip(records, scores):
            cluster_membership[record_id] = {
                "Cluster ID": cluster_id,
                "confidence_score": score,
            }

    with open(output_file, "w") as f_output:
        fieldnames = ["id", "company_name", "Cluster ID", "confidence_score"]
        writer = csv.DictWriter(f_output, fieldnames=fieldnames)
        writer.writeheader()  # # TypeError: 'int' object is not subscriptable

        for row in data_d.values():
            row_id = int(row["id"])
            row.update({"id": row_id, **cluster_membership[row_id]})
            writer.writerow(row)


def dedup_companies():
    query = """
        select lower(company_name) as company_name from (
            select distinct sponsor as company_name from trials

            UNION ALL

            select assignee.name as company_name
            FROM applications a,
            unnest(a.assignees) as assignee
        ) s
        group by lower(company_name)
        having count(*) > 100
    """
    records = PsqlDatabaseClient().select(query)
    company_map = dict([(i, {**record, "id": i}) for i, record in enumerate(records)])
    variables = [
        {"field": "company_name", "type": "String"},
    ]
    dedup_interactive(company_map, variables)


def persist_synonyms(filename: str = COMPANY_DEDUP_FILE):
    reader = csv.DictReader(open(filename))
    records = [row for row in reader]

    lookup_map = dict([(record["id"], record["company_name"]) for record in records])

    synonym_map_records = [
        {"synonym": record["company_name"], "term": lookup_map[record["id"]]}
        for record in records
        if record["Cluster ID"] != record["id"]
    ]
    records = PsqlDatabaseClient().insert_into_table(synonym_map_records, "synonym_map")


def main():
    """
    Dedup
    """
    dedup_companies()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.dedup_companies")
        sys.exit()

    main()
