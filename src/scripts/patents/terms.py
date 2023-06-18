"""
Functions to initialize the terms and synonym tables
"""
from itertools import groupby
from typing import TypedDict
import time
from google.cloud import bigquery
import logging

from clients.low_level.big_query import select_from_bg
from clients.low_level.big_query import (
    BQ_DATASET_ID,
    BQ_DATASET,
)
from common.ner import TermNormalizer, NormalizationMap
from common.utils.list import batch, dedup

from .local_constants import SYNONYM_MAP

TermRecord = TypedDict(
    "TermRecord",
    {
        "term": str,
        "count": int,
        "canonical_id": str,
        "domains": list[str],
        "original_terms": list[str],
        "original_ids": list[str],
    },
)


def __get_normalized_terms(rows):
    """
    Normalizes terms based on canonical matches; dedups.

    Args:
        rows: list of rows from the terms query
        normalization_map: map of term to canonical term
    """
    normalizer = TermNormalizer()
    normalization_map = normalizer.generate_map([row["term"] for row in rows])

    def __get_term(row):
        entry = normalization_map.get(row["term"])
        if not entry:
            return SYNONYM_MAP.get(row["term"].lower()) or row["term"]
        return entry.canonical_name

    def __get_term_record(_group) -> TermRecord:
        group = list(_group)
        return {
            "term": [row["term"] for row in group][0],  # non-lowered-term
            "count": sum(row["count"] for row in group),
            "canonical_id": group[0].get("canonical_id") or "",
            "domains": dedup([row["domain"] for row in group]),
            "original_terms": dedup([row["original_term"] for row in group]),
            "original_ids": dedup([row["original_id"] for row in group]),
        }

    normalized_terms = [
        {
            "term": __get_term(row),
            "count": row["count"] or 0,
            "canonical_id": getattr(
                normalization_map.get(row["term"]) or (), "concept_id", None
            ),
            "domain": row["domain"],
            "original_term": row["term"],
            "original_id": row["ocid"],
        }
        for row in rows
    ]
    # sort required for groupby
    sorted_terms = sorted(normalized_terms, key=lambda row: row["term"].lower())
    grouped_terms = groupby(sorted_terms, key=lambda row: row["term"].lower())

    terms = [__get_term_record(group) for _, group in grouped_terms]
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

    # Normalize, dedupe, and count the terms
    normalized_terms = __get_normalized_terms(rows)

    # Create a new table to hold the modified records
    new_table = bigquery.Table(f"{BQ_DATASET_ID}.terms")
    new_table.schema = [
        bigquery.SchemaField("term", "STRING"),
        bigquery.SchemaField("canonical_id", "STRING"),
        bigquery.SchemaField("count", "INTEGER"),
        bigquery.SchemaField("domains", "STRING", mode="REPEATED"),
        bigquery.SchemaField("original_terms", "STRING", mode="REPEATED"),
        bigquery.SchemaField("original_ids", "STRING", mode="REPEATED"),
    ]
    new_table = client.create_table(new_table, exists_ok=True)
    time.sleep(15)  # wait. (TODO: should check for existence instead)

    batched = batch(normalized_terms)
    for b in batched:
        client.insert_rows(new_table, b)
        logging.info(f"Inserted %s rows into terms table", len(b))

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
    logging.info("Creating synonym map")
    client = bigquery.Client()
    new_table = bigquery.Table(f"{BQ_DATASET_ID}.synonym_map")
    new_table.schema = [
        bigquery.SchemaField("synonym", "STRING"),
        bigquery.SchemaField("term", "INTEGER"),
    ]
    new_table = client.create_table(new_table, exists_ok=True)
    time.sleep(15)  # wait. (TODO: should check for existence instead)

    logging.info("Adding default synonym map entries")
    __add_to_synonym_map(synonym_map)


def __add_to_synonym_map(synonym_map: dict[str, str]):
    """
    Add common entity names to the synonym map

    Args:
        synonym_map: a map of synonyms to canonical names
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
        logging.info("Inserted %s rows into synonym_map (errors: %s)", len(b), errors)


def create_patent_terms():
    """
    Create the terms and synonym map tables
    """
    __create_synonym_map(SYNONYM_MAP)
    __create_terms()
