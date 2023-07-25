"""
Functions to initialize the terms and synonym tables
"""
from itertools import groupby
from typing import Optional, TypedDict
import time
from google.cloud import bigquery
import logging

from clients.low_level.big_query import select_from_bg
from clients.low_level.big_query import (
    BQ_DATASET_ID,
    BQ_DATASET,
)
from common.ner import TermLinker
from common.utils.list import batch, dedup
from clients.low_level.big_query import execute_with_retries
from clients.patents.utils import clean_assignee

from ._constants import BIOSYM_ANNOTATIONS_TABLE, SYNONYM_MAP

MIN_ASSIGNEE_COUNT = 10

BaseTermRecord = TypedDict(
    "BaseTermRecord", {"term": str, "count": int, "canonical_id": Optional[str]}
)


class TermRecord(BaseTermRecord):
    domain: str
    original_term: str
    original_id: Optional[str]


class AggregatedTermRecord(BaseTermRecord):
    domains: list[str]
    synonyms: list[str]
    synonym_ids: list[str]


def __get_terms():
    """
    Collects all terms for the terms table
    From
    - gpr_annotations
    - biosym_annotations
    - publications (assignee_harmonized)
    """

    def __aggregate_terms(terms: list[TermRecord]) -> list[AggregatedTermRecord]:
        """
        Aggregates terms by term and canonical id
        """
        # sort required for groupby
        sorted_terms = sorted(terms, key=lambda row: row["term"].lower())
        grouped_terms = groupby(sorted_terms, key=lambda row: row["term"].lower())

        def __get_term_record(_group) -> AggregatedTermRecord:
            group = list(_group)
            return {
                "term": group[0]["term"],  # non-lowered-term
                "count": sum(row["count"] for row in group),
                "canonical_id": group[0].get("canonical_id") or "",
                "domains": dedup([row["domain"] for row in group]),
                "synonyms": dedup([row["original_term"] for row in group]),
                "synonym_ids": dedup([row.get("original_id") for row in group]),
            }

        agg_terms = [__get_term_record(group) for _, group in grouped_terms]
        return agg_terms

    def __get_assignee_terms() -> list[AggregatedTermRecord]:
        """
        Creates assignee terms from the publications table
        """
        assignees_query = f"""
            SELECT assignee.name as assignee, "assignee" as domain, count(*) as count
            FROM `{BQ_DATASET_ID}.publications` p,
            unnest(p.assignee_harmonized) as assignee
            group by assignee
        """
        rows = select_from_bg(assignees_query)
        normalized: list[TermRecord] = [
            {
                "term": clean_assignee(row["assignee"]),  # normalized assignee name
                "count": row["count"] or 0,
                "domain": row["domain"],
                "canonical_id": None,
                "original_term": row["assignee"],
                "original_id": None,
            }
            for row in rows
        ]

        terms = __aggregate_terms(normalized)
        return [term for term in terms if term["count"] > MIN_ASSIGNEE_COUNT]

    def __get_entity_terms() -> list[AggregatedTermRecord]:
        """
        Creates entity terms from the annotations table
        Normalizes terms and associates to canonical ids
        """
        terms_query = f"""
            SELECT term, original_id, domain, COUNT(*) as count
            FROM
            (
                -- gpr annotations
                SELECT preferred_name as term, CAST(ocid as STRING) as original_id, domain
                FROM `{BQ_DATASET_ID}.gpr_annotations`
                where length(preferred_name) > 1

                UNION ALL

                -- biosym annotations
                SELECT original_term as term, canonical_id as original_id, domain
                FROM `{BQ_DATASET_ID}.{BIOSYM_ANNOTATIONS_TABLE}`
                where length(original_term) > 1
            ) AS all_annotations
            group by term, original_id, domain
        """
        rows = select_from_bg(terms_query)

        linker = TermLinker()
        normalization_map = dict(linker([row["term"] for row in rows]))

        def __normalize(row):
            entry = normalization_map.get(row["term"])
            if not entry:
                return SYNONYM_MAP.get(row["term"].lower()) or row["term"]
            return entry.name

        terms: list[TermRecord] = [
            {
                "term": __normalize(row),
                "count": row["count"] or 0,
                "canonical_id": getattr(
                    normalization_map.get(row["term"]) or (), "concept_id", None
                ),
                "domain": row["domain"],
                "original_term": row["term"],
                "original_id": row["original_id"],
            }
            for row in rows
        ]

        return __aggregate_terms(terms)

    # Normalize, dedupe, and count the terms
    entity_terms = __get_entity_terms()
    assignee_terms = __get_assignee_terms()

    terms = assignee_terms + entity_terms
    return terms


def __create_terms():
    """
    Create a table of entities

    - pulls distinct annotation and assigee values
    - normalizes the terms
    - inserts them into a new table
    - adds synonym entries for the original terms
    """
    client = bigquery.Client()

    # Create a new table to hold the modified records
    table_id = f"{BQ_DATASET_ID}.terms"
    new_table = bigquery.Table(table_id)
    new_table.schema = [
        bigquery.SchemaField("term", "STRING"),
        bigquery.SchemaField("canonical_id", "STRING"),
        bigquery.SchemaField("count", "INTEGER"),
        bigquery.SchemaField("domains", "STRING", mode="REPEATED"),
        bigquery.SchemaField("synonyms", "STRING", mode="REPEATED"),
        bigquery.SchemaField("synonym_ids", "STRING", mode="REPEATED"),
    ]
    client.delete_table(table_id, not_found_ok=True)
    new_table = client.create_table(new_table)

    # grab terms from annotation tables (slow!!)
    terms = __get_terms()

    batched = batch(terms)
    for b in batched:
        execute_with_retries(lambda: client.insert_rows(new_table, b))
        logging.info(f"Inserted %s rows into terms table", len(b))

    # Persist term -> synonyms as synonyms
    synonym_map = {og_term: row["term"] for row in terms for og_term in row["synonyms"]}

    execute_with_retries(lambda: __add_to_synonym_map(synonym_map))


def __init_synonym_map(synonym_map: dict[str, str]):
    """
    Create a table of synonyms

    Args:
        synonym_map: a map of synonyms to canonical names
    """
    logging.info("Creating synonym map")
    client = bigquery.Client()
    table_id = f"{BQ_DATASET_ID}.synonym_map"
    table = bigquery.Table(table_id)
    table.schema = [
        bigquery.SchemaField("synonym", "STRING"),
        bigquery.SchemaField("term", "STRING"),
    ]

    # remove and (re)create the table
    client.delete_table(table_id, not_found_ok=True)
    table = client.create_table(table)

    logging.info("Adding default/hard-coded synonym map entries")
    execute_with_retries(lambda: __add_to_synonym_map(synonym_map))


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

    Idempotent (all tables are dropped and recreated)
    """
    __init_synonym_map(SYNONYM_MAP)
    __create_terms()
