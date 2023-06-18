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
from common.ner import TermNormalizer
from common.utils.list import batch, dedup
from src.clients.patents.utils import clean_assignee

from .local_constants import SYNONYM_MAP

BaseTermRecord = TypedDict(
    "BaseTermRecord", {"term": str, "count": int, "canonical_id": Optional[str]}
)


class TermRecord(BaseTermRecord):
    domain: str
    original_term: str
    original_id: Optional[str]


class AggregatedTermRecord(BaseTermRecord):
    domains: list[str]
    original_terms: list[str]
    original_ids: list[str]


def __aggregate_terms(terms: list[TermRecord]) -> list[AggregatedTermRecord]:
    """
    Aggregates terms by term and canonical id

    Args:
        terms (list[TermRecord]): list of normalized term records (potentially with dup names)
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
            "original_terms": dedup([row["original_term"] for row in group]),
            "original_ids": dedup([row["id"] for row in group]),
        }

    agg_terms = [__get_term_record(group) for _, group in grouped_terms]
    return agg_terms


def __get_entity_terms() -> list[AggregatedTermRecord]:
    """
    Creates entity terms from the annotations table
    Normalizes terms and associates to canonical ids

    Args:
        rows: list of rows from the terms query
    """
    terms_query = f"""
        SELECT preferred_name as term, ocid as id, count(*) as count, domain
        FROM `{BQ_DATASET_ID}.gpr_annotations`
        group by preferred_name, ocid, domain
    """
    rows = select_from_bg(terms_query)

    normalizer = TermNormalizer()
    normalization_map = normalizer.generate_map([row["term"] for row in rows])

    def __get_term(row):
        entry = normalization_map.get(row["term"])
        if not entry:
            return SYNONYM_MAP.get(row["term"].lower()) or row["term"]
        return entry.canonical_name

    terms: list[TermRecord] = [
        {
            "term": __get_term(row),
            "count": row["count"] or 0,
            "canonical_id": getattr(
                normalization_map.get(row["term"]) or (), "concept_id", None
            ),
            "domain": row["domain"],
            "original_term": row["term"],
            "original_id": row["id"],
        }
        for row in rows
    ]

    return __aggregate_terms(terms)


def __get_assignee_terms() -> list[AggregatedTermRecord]:
    """
    Creates assignee terms from the publications table
    """
    assignees_query = f"""
        SELECT assignee.name as assignee, count(*) as count, "assignee" as domain
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
            "original_term": row["term"],
            "original_id": None,
        }
        for row in rows
    ]

    terms = __aggregate_terms(normalized)
    return terms


def __create_terms():
    """
    Create a table of entities

    - pulls distinct from the gpr_annotations table
    - normalizes the terms
    - inserts them into a new table
    """
    client = bigquery.Client()

    # Normalize, dedupe, and count the terms
    entity_terms = __get_entity_terms()
    assignee_terms = __get_assignee_terms()

    terms = assignee_terms + entity_terms

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

    batched = batch(terms)
    for b in batched:
        client.insert_rows(new_table, b)
        logging.info(f"Inserted %s rows into terms table", len(b))

    # Persist term -> original_terms as synonyms
    synonym_map = {
        og_term: row["term"] for row in terms for og_term in row["original_terms"]
    }

    __add_to_synonym_map(synonym_map)


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
