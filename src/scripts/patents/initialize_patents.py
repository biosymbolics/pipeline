"""
Functions to initialize the patents database
"""
from itertools import groupby
import time
from google.cloud import bigquery
import logging

from system import initialize

initialize()

from clients.low_level.big_query import select_from_bg
from clients.low_level.big_query import (
    execute_bg_query,
    query_to_bg_table,
    BQ_DATASET_ID,
    BQ_DATASET,
)
from common.ner import TermNormalizer, NormalizationMap
from common.utils.list import batch, dedup

from .copy import copy_patent_tables
from .local_constants import SYNONYM_MAP


logging.basicConfig(level=logging.INFO)


def __get_normalized_terms(rows, normalization_map: NormalizationMap):
    """
    Normalizes terms based on canonical matches; dedups.
    """

    def __get_term(row):
        entry = normalization_map.get(row["term"])
        if not entry:
            return SYNONYM_MAP.get(row["term"].lower()) or row["term"]
        return entry.canonical_name

    normalized_terms = [
        {
            "term": __get_term(row),
            "count": row["count"] or 0,
            "canonical_id": getattr(
                normalization_map.get(row["term"]) or (), "concept_id", None
            ),
            "original_term": row["term"],
            "original_id": row["ocid"],
        }
        for row in rows
    ]
    # sort required for groupby
    sorted_terms = sorted(normalized_terms, key=lambda row: row["term"].lower())
    grouped_terms = groupby(sorted_terms, key=lambda row: row["term"].lower())

    def __get_term_obj(_group):
        group = list(_group)
        return {
            "term": [row["term"] for row in group][0],  # non-lowered-term
            "count": sum(row["count"] for row in group),
            "canonical_id": dedup([row["canonical_id"] for row in group]),
            "original_terms": dedup([row["original_term"] for row in group]),
            "original_ids": dedup([row["original_id"] for row in group]),
        }

    terms = [__get_term_obj(group) for _, group in grouped_terms]
    return terms


def __create_terms():
    """
    Create a table of entities

    - pulls distinct from the gpr_annotations table
    - normalizes the terms
    - inserts them into a new table
    """
    client = bigquery.Client()

    terms_query = f"""
        SELECT preferred_name as term, ocid, count(*) as count
        FROM `{BQ_DATASET_ID}.gpr_annotations`
        group by preferred_name, ocid
    """
    rows = select_from_bg(terms_query)

    normalizer = TermNormalizer()
    normalization_map = normalizer.generate_map([row["term"] for row in rows])

    # Normalize, dedupe, and count the terms
    normalized_terms = __get_normalized_terms(rows, normalization_map)

    # Create a new table to hold the modified records
    # TODO: wait after create??
    new_table = bigquery.Table(f"{BQ_DATASET_ID}.terms")
    new_table.schema = [
        bigquery.SchemaField("term", "STRING"),
        bigquery.SchemaField("count", "INTEGER"),
        bigquery.SchemaField("domain", "STRING"),
        bigquery.SchemaField("canonical_id", "STRING", mode="REPEATED"),
        bigquery.SchemaField("original_terms", "STRING", mode="REPEATED"),
        bigquery.SchemaField("original_ids", "STRING", mode="REPEATED"),
    ]
    new_table = client.create_table(new_table, exists_ok=True)
    time.sleep(15)  # wait. silly.

    batched = batch(normalized_terms)
    for b in batched:
        client.insert_rows(new_table, b)
        logging.info(f"Inserted {len(b)} rows")

    syn_map = {
        og_term: row["term"]
        for row in normalized_terms
        for og_term in row["original_terms"]
    }

    __add_to_synonym_map(syn_map)


def __create_synonym_map(synonym_map: dict[str, str]):
    """
    Create a table of synonyms

    Args:
        synonym_map: a map of synonyms to canonical names
    """
    create = "CREATE TABLE patents.synonym_map (synonym STRING, term STRING);"
    execute_bg_query(create)
    __add_to_synonym_map(synonym_map)


def __add_to_synonym_map(synonym_map: dict[str, str]):
    """
    Add common entity names to the synonym map
    """
    client = bigquery.Client()

    data = [
        {
            "synonym": entry[0].lower() if entry[0] is not None else None,
            "term": entry[1].lower(),
        }
        for entry in synonym_map.items()
        if entry[1] is not None and entry[0] != entry[1]
    ]
    batched = batch(data)

    for b in batched:
        table_ref = client.dataset(BQ_DATASET).table("synonym_map")
        errors = client.insert_rows_json(table_ref, b)
        logging.info("Inserted %s rows (errors: %s)", len(b), errors)


FIELDS = [
    # gpr_publications
    "gpr_pubs.publication_number as publication_number",
    "abstract",
    "application_number",
    "cited_by",
    "country",
    "embedding_v1 as embeddings",
    "similar",
    "title",
    "top_terms",
    "url",
    # publications
    "application_kind",
    "assignee as assignee_raw",
    "assignee_harmonized",
    "ARRAY(SELECT assignee.name FROM UNNEST(assignee_harmonized) as assignee) as assignees",
    "citation",
    "claims_localized as claims",
    "ARRAY(SELECT cpc.code FROM UNNEST(pubs.cpc) as cpc) as cpc_codes",
    "family_id",
    "filing_date",
    "grant_date",
    "inventor as inventor_raw",
    "inventor_harmonized",
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


def __create_applications_table():
    """
    Create a table of patent applications for use in app queries
    """
    logging.info("Create a table of patent applications for use in app queries")
    applications = f"""
        SELECT "
        {','.join(FIELDS)}
        FROM `{BQ_DATASET_ID}.publications` as pubs,
        `{BQ_DATASET_ID}.gpr_publications` as gpr_pubs
        WHERE pubs.publication_number = gpr_pubs.publication_number
    """
    query_to_bg_table(applications, "applications")


def __create_annotations_table():
    """
    Create a table of annotations for use in app queries
    """
    logging.info("Create a table of annotations for use in app queries")

    entity_query = f"""
        WITH ranked_terms AS (
            SELECT
                publication_number,
                ocid,
                LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) as term,
                domain,
                confidence,
                source,
                character_offset_start,
                ROW_NUMBER() OVER(PARTITION BY publication_number, LOWER(IF(map.term IS NOT NULL, map.term, a.preferred_name)) ORDER BY character_offset_start) as rank
            FROM `{BQ_DATASET_ID}.gpr_annotations` a
            LEFT JOIN `{BQ_DATASET_ID}.synonym_map` map ON LOWER(a.preferred_name) = map.synonym
        )
        SELECT
            publication_number,
            ARRAY_AGG(
                struct(ocid, term, domain, confidence, source, character_offset_start)
                ORDER BY character_offset_start
            ) as annotations
        FROM ranked_terms
        WHERE rank = 1
        GROUP BY publication_number
    """
    query_to_bg_table(entity_query, "annotations")


def main():
    """
    Copy tables from patents-public-data to a local dataset

    Order matters. Non-idempotent.
    """
    # copy gpr_publications, publications, gpr_annotations tables
    # copy_patent_tables()

    # create synonym_map table (for final entity names)
    __create_synonym_map(SYNONYM_MAP)

    # create terms table and update synonym map
    __create_terms()

    # create the (small) tables against which the app will query
    # __create_applications_table()
    __create_annotations_table()


if __name__ == "__main__":
    main()
