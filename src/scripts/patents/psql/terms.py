"""
Functions to initialize the terms and synonym tables
"""
from itertools import groupby
from typing import Optional, TypedDict
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import WORKING_BIOSYM_ANNOTATIONS_TABLE
from data.ner import TermNormalizer
from utils.file import load_json_from_file, save_json_as_file
from utils.list import dedup

from .biosym_annotations import populate_working_biosym_annotations

from .._constants import SYNONYM_MAP
from ..utils import clean_assignees

TERMS_FILE = "terms.json"


class BaseTermRecord(TypedDict):
    term: str
    count: int
    canonical_id: Optional[str]


class TermRecord(BaseTermRecord):
    domain: str
    original_term: str
    original_id: Optional[str]


class AggregatedTermRecord(BaseTermRecord):
    domains: list[str]
    synonyms: list[str]
    synonym_ids: list[str]


SYNONYM_TABLE_NAME = "synonym_map"


class SynonymMapper:
    """
    Static class for synonym mapping functions
    """

    def __init__(self, synonym_map: dict[str, str]):
        """
        Create a table of synonyms

        Args:
            synonym_map: a map of synonyms to canonical names
        """
        self.client = PsqlDatabaseClient()

        logging.info("Creating synonym map (truncating if exists)")
        self.client.create_table(
            SYNONYM_TABLE_NAME,
            {"synonym": "TEXT", "term": "TEXT"},
            exists_ok=True,
            truncate_if_exists=True,
        )

        logging.info("Adding default/hard-coded synonym map entries")
        self.add_map(synonym_map)

    def add_synonyms(
        self,
        existing_terms: Optional[list[AggregatedTermRecord]] = None,
    ):
        """
        Add synonym records based on terms, i.e. records with { term: "foo", synonyms: ["bar", "baz"] }

        Args:
            existing_terms: a list of terms to use as the basis for synonyms.
                If not provided, terms will be loaded from file (TERMS_FILE/terms.json)
        """
        terms = existing_terms or load_json_from_file(TERMS_FILE)

        if not isinstance(terms, list):
            logging.error("Terms must be a list, instead is type %s", type(terms))
            raise Exception("Terms must be a list")

        # Persist term -> synonyms as synonyms
        synonym_map = {
            og_term: row["term"]
            for row in terms
            for og_term in row["synonyms"]
            if len(row["term"]) > 1
        }

        logging.info("Adding %s terms to synonym map", len(synonym_map))

        self.add_map(synonym_map)

    def add_map(self, synonym_map: dict[str, str]):
        """
        Add common entity names to the synonym map, taking in a map of form {synonym: term}

        Args:
            synonym_map: a map of synonyms to canonical names
        """
        records = [
            {
                "synonym": entry[0].lower(),
                "term": entry[1].lower(),
            }
            for entry in synonym_map.items()
            if len(entry[1]) > 0 and len(entry[0]) > 0 and entry[0] != entry[1]
        ]

        self.client.insert_into_table(records, SYNONYM_TABLE_NAME)
        logging.debug(
            "Inserted %s rows into synonym_map (%s)",
            len(records),
            len(list(synonym_map.keys())),
        )


class TermAssembler:
    """
    Static class for assembling terms and synonyms
    """

    def __init__(self):
        self.client = PsqlDatabaseClient()

    @staticmethod
    def __aggregate(terms: list[TermRecord]) -> list[AggregatedTermRecord]:
        """
        Aggregates terms by term and canonical id
        """
        # sort required for groupby
        sorted_terms = sorted(terms, key=lambda row: row["term"].lower())
        grouped_terms = groupby(sorted_terms, key=lambda row: row["term"].lower())

        def __get_term_record(_group) -> AggregatedTermRecord:
            group = list(_group)
            return {
                "term": group[0]["term"],
                "count": sum(row["count"] for row in group),
                "canonical_id": group[0].get("canonical_id") or "",
                "domains": dedup([row["domain"] for row in group]),
                "synonyms": dedup([row["original_term"] for row in group]),
                "synonym_ids": dedup([row.get("original_id") for row in group]),
            }

        agg_terms = [__get_term_record(group) for _, group in grouped_terms]
        return agg_terms

    def __generate_owner_terms(self) -> list[AggregatedTermRecord]:
        """
        Generates owner terms (assignee/inventor) from the applications table
        """
        owner_query = f"""
            SELECT unnest(assignees) as name, 'assignees' as domain, count(*) as count
            FROM applications a
            group by name

            UNION ALL

            SELECT unnest(inventors) as name, 'inventors' as domain, count(*) as count
            FROM applications a
            group by name
        """
        rows = self.client.select(owner_query)
        names = [row["name"] for row in rows]
        cleaned = clean_assignees(names)

        normalized: list[TermRecord] = [
            {
                "term": assignee if row["domain"] == "assignees" else row["name"],
                "count": row["count"] or 0,
                "domain": row["domain"],
                "canonical_id": None,
                "original_term": row["name"],
                "original_id": None,
            }
            for row, assignee in zip(rows, cleaned)
        ]

        terms = TermAssembler.__aggregate(
            [row for row in normalized if len(row["term"]) > 1]
        )
        return terms

    def __generate_entity_terms(self) -> list[AggregatedTermRecord]:
        """
        Creates entity terms from the annotations table
        Normalizes terms and associates to canonical ids
        """
        terms_query = f"""
                SELECT lower(original_term) as term, domain, COUNT(*) as count
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
                where length(original_term) > 1
                group by lower(original_term), domain
            """
        rows = self.client.select(terms_query)
        terms: list[str] = [row["term"] for row in rows]

        logging.info("Normalizing/linking %s terms", len(rows))
        normalizer = TermNormalizer()
        normalization_map = dict(normalizer.normalize(terms))

        logging.info("Finished creating normalization_map")

        def __normalize(term: str, domain: str) -> str:
            if domain == "attributes":
                return term  # leave attributes alone
            entry = normalization_map.get(term)
            if not entry:
                return SYNONYM_MAP.get(term) or term
            return entry.name

        term_records: list[TermRecord] = [
            {
                "term": __normalize(row["term"], row["domain"]),
                "count": row["count"] or 0,
                "canonical_id": getattr(
                    normalization_map.get(row["term"]) or (), "concept_id", None
                ),
                "domain": row["domain"],
                "original_term": row["term"],
                "original_id": "",  # row["original_id"],
            }
            for row in rows
        ]

        return TermAssembler.__aggregate(term_records)

    def generate_terms(self):
        """
        Collects and forms terms for the terms table
        From
        - biosym_annotations
        - applications (assignees + inventors)
        """
        logging.info("Getting owner (assignee, inventor) terms")
        assignee_terms = self.__generate_owner_terms()

        logging.info("Generating entity terms")
        entity_terms = self.__generate_entity_terms()

        terms = assignee_terms + entity_terms
        return terms

    def create_and_persist_terms(self):
        """
        Extracts/generates terms and persists them to a table

        - pulls distinct annotation and assigee values
        - normalizes the terms
        - inserts them into a new table
        - adds synonym entries for the original terms
        """
        table_name = "terms"
        schema = {
            "term": "TEXT",
            "canonical_id": "TEXT",
            "count": "INTEGER",
            "domains": "TEXT[]",
            "synonyms": "TEXT[]",
            "synonym_ids": "TEXT[]",
        }
        self.client.create_table(
            table_name, schema, exists_ok=True, truncate_if_exists=True
        )

        # grab terms from annotation tables (slow!!)
        terms = self.generate_terms()
        save_json_as_file(terms, TERMS_FILE)

        self.client.insert_into_table(terms, table_name)
        self.client.create_indices(
            [
                {
                    "table": table_name,
                    "column": "term",
                    "is_trgm": True,
                },
            ]
        )
        logging.info(f"Inserted %s rows into terms table", len(terms))

    @staticmethod
    def run():
        """
        Create the terms and synonym map tables

        Idempotent (all tables are dropped and recreated)
        """
        sm = SynonymMapper(SYNONYM_MAP)
        TermAssembler().create_and_persist_terms()
        sm.add_synonyms()


def create_patent_terms():
    """
    Create the terms and synonym map tables

    Idempotent (all tables are dropped and recreated)
    """
    populate_working_biosym_annotations()

    TermAssembler.run()
