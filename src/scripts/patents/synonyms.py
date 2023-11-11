"""
Functions to initialize the terms and synonym tables
"""
from typing import Optional
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from utils.file import load_json_from_file

from .constants import SYNONYM_TABLE_NAME, TERMS_FILE
from .types import AggregatedTermRecord


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
            if len(entry[0]) > 0 and entry[0] != entry[1]
        ]

        self.client.insert_into_table(records, SYNONYM_TABLE_NAME)
        logging.info(
            "Inserted %s rows into synonym_map (%s)",
            len(records),
            len(list(synonym_map.keys())),
        )

    def index(self):
        self.client.create_index({"table": "synonym_map", "column": "synonym"})
