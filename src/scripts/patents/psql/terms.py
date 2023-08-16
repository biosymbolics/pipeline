"""
Functions to initialize the terms and synonym tables
"""
from itertools import groupby
from typing import Optional, TypedDict
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.database import execute_with_retries
from data.ner import TermNormalizer
from utils.file import load_json_from_file, save_json_as_file
from utils.list import dedup


from .biosym_annotations import create_working_biosym_annotations

from .._constants import BIOSYM_ANNOTATIONS_TABLE
from ..bq.gpr_constants import SYNONYM_MAP
from ..utils import clean_assignees

MIN_ASSIGNEE_COUNT = 5
TERMS_FILE = "terms.json"

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
        self.client.truncate_table(SYNONYM_TABLE_NAME)
        self.client.create_table(
            SYNONYM_TABLE_NAME, {"synonym": "TEXT", "term": "TEXT"}
        )

        logging.info("Adding default/hard-coded synonym map entries")
        execute_with_retries(lambda: self.add_map(synonym_map))

    def add_terms(
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

        self.add_map(synonym_map)

    def add_map(self, synonym_map: dict[str, str]):
        """
        Add common entity names to the synonym map, taking in a map of form {synonym: term}

        Args:
            synonym_map: a map of synonyms to canonical names
        """
        data = [
            {
                "synonym": entry[0].lower(),
                "term": entry[1].lower(),
            }
            for entry in synonym_map.items()
            if entry[1] is not None and entry[0] is not None and entry[0] != entry[1]
        ]

        self.client.insert_into_table(data, SYNONYM_TABLE_NAME)
        logging.info("Inserted %s rows into synonym_map", len(data))


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
                "term": group[0]["term"],  # non-lowered-term
                "count": sum(row["count"] for row in group),
                "canonical_id": group[0].get("canonical_id") or "",
                "domains": dedup([row["domain"] for row in group]),
                "synonyms": dedup([row["original_term"] for row in group]),
                "synonym_ids": dedup([row.get("original_id") for row in group]),
            }

        agg_terms = [__get_term_record(group) for _, group in grouped_terms]
        return agg_terms

    def __fetch_owner_terms(self) -> list[AggregatedTermRecord]:
        """
        Creates owner terms (assignee/inventor) from the applications table
        """
        owner_query = f"""
            SELECT assignee as name, "assignee" as domain, count(*) as count
            FROM applications a,
            unnest(a.assignee_harmonized) as assignee
            group by name

            UNION ALL

            SELECT inventor as name, "inventor" as domain, count(*) as count
            FROM  applications a,
            unnest(a.inventor_harmonized) as inventor
            group by name
        """
        rows = self.client.select(owner_query)
        names = [row["name"] for row in rows]
        cleaned = clean_assignees(names)

        normalized: list[TermRecord] = [
            {
                "term": assignee if row["domain"] == "assignee" else row["name"],
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
        return [term for term in terms if term["count"] > MIN_ASSIGNEE_COUNT]

    def __generate_entity_terms(self) -> list[AggregatedTermRecord]:
        """
        Creates entity terms from the annotations table
        Normalizes terms and associates to canonical ids
        """
        terms_query = f"""
                SELECT term, original_id, domain, COUNT(*) as count
                FROM
                (
                    SELECT original_term as term, canonical_id as original_id, domain
                    FROM {BIOSYM_ANNOTATIONS_TABLE}
                    where length(original_term) > 1
                ) AS all_annotations
                group by lower(term), original_id, domain
            """
        rows = self.client.select(terms_query)
        terms: list[str] = [row["term"] for row in rows]

        logging.info("Normalizing/linking %s terms", len(rows))
        normalizer = TermNormalizer()
        normalization_map = dict(normalizer.normalize(terms))

        logging.info("Finished creating normalization_map")

        def __normalize(row):
            entry = normalization_map.get(row["term"])
            if not entry:
                return SYNONYM_MAP.get(row["term"]) or row["term"]
            return entry.name

        term_records: list[TermRecord] = [
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

        return TermAssembler.__aggregate(term_records)

    def generate_terms(self):
        """
        Collects and forms terms for the terms table
        From
        - biosym_annotations
        - applications (assignees + inventors)
        """
        logging.info("Getting owner (assignee, inventor) terms")
        assignee_terms = self.__fetch_owner_terms()

        logging.info("Generating entity terms")
        entity_terms = self.__generate_entity_terms()

        terms = assignee_terms + entity_terms
        return terms

    def create_terms(self):
        """
        Fetches terms and persists them to a table

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

        rows = [row for row in terms if len(row["term"]) > 1]
        self.client.insert_into_table(rows, table_name)
        logging.info(f"Inserted %s rows into terms table", len(rows))

    @staticmethod
    def run():
        """
        Create the terms and synonym map tables

        Idempotent (all tables are dropped and recreated)
        """
        sm = SynonymMapper(SYNONYM_MAP)
        TermAssembler().create_terms()
        sm.add_terms()


def create_patent_terms():
    """
    Create the terms and synonym map tables

    Idempotent (all tables are dropped and recreated)
    """
    create_working_biosym_annotations()

    TermAssembler.run()
