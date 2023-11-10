"""
Functions to initialize the terms and synonym tables
"""
import sys
from typing import Optional, Sequence, TypedDict
import logging
from pydash import compact, flatten, group_by
from core.ner.utils import lemmatize_tails

import system

system.initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import WORKING_BIOSYM_ANNOTATIONS_TABLE
from core.ner import TermNormalizer
from utils.file import load_json_from_file, save_json_as_file
from utils.list import dedup

from .constants import SYNONYM_MAP
from .process_biosym_annotations import populate_working_biosym_annotations
from .utils import clean_owners

TERMS_FILE = "terms.json"
TERMS_TABLE = "terms"
MIN_CANONICAL_NAME_COUNT = 5


class BaseTermRecord(TypedDict):
    term: str
    count: int
    id: Optional[str]
    ids: Optional[list[str]]


class TermRecord(BaseTermRecord):
    domain: str
    original_term: Optional[str]


class AggregatedTermRecord(BaseTermRecord):
    domains: list[str]
    synonyms: list[str]


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


class TermAssembler:
    """
    Static class for assembling terms and synonyms

    Terms assembled:
    - owner/sponsor/applications (companies, universities and individuals)
    - entities/annotations (compounds, diseases, mechanisms)
    """

    def __init__(self):
        self.client = PsqlDatabaseClient()

    @staticmethod
    def _get_canonical_name(term_records: Sequence[TermRecord]) -> str:
        """
        Get canonical name from a list of term records

        Uses the most common "original_term" if used with sufficiently high frequency,
        and otherwise uses the "canonical_term" (which is often a mangled composite of UMLS terms)
        """
        synonyms = sorted(
            group_by(term_records, "original_term").items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )
        return (
            synonyms[0][0]
            if len(synonyms[0][1]) > MIN_CANONICAL_NAME_COUNT
            else term_records[0]["term"]
        )

    @staticmethod
    def __group_terms(terms: list[TermRecord]) -> list[AggregatedTermRecord]:
        """
        Dedups/groups terms by id and term
        """
        # depends upon id always being in the same order (enforced elsewhere)
        grouped_terms: dict[str, Sequence[TermRecord]] = group_by(
            [{**t, "key": t["id"] or t["term"]} for t in terms],  # type: ignore
            "key",
        )

        def __get_term_record(group: Sequence[TermRecord]) -> AggregatedTermRecord:
            canonical_term = TermAssembler._get_canonical_name(group)
            return {
                "term": canonical_term,
                "count": sum(row["count"] for row in group),
                "id": group[0].get("id") or "",
                "ids": group[0].get("ids") or [],
                "domains": dedup([row["domain"] for row in group]),
                # lemmatize tails for less duplication. todo: lemmatize all?
                # 2x duplication for perf
                "synonyms": dedup(
                    lemmatize_tails(
                        dedup([row["original_term"] or "" for row in group])
                    )
                ),
            }

        agg_terms = [__get_term_record(group) for _, group in grouped_terms.items()]
        return agg_terms

    def generate_owner_terms(self) -> list[AggregatedTermRecord]:
        """
        Generates owner terms (assignee/inventor) from:
        - patent applications table
        - aact (ctgov)
        - drugcentral approvals

        TODO:
        National Cancer Center, boston therapeutics
        """
        db_owner_query_map = {
            # patents db
            "patents": """
                SELECT lower(unnest(assignees)) as name, 'assignees' as domain, count(*) as count
                FROM applications a
                group by name
                having count(*) > 20 -- individuals unlikely to have more patents

                UNION ALL

                SELECT lower(unnest(inventors)) as name, 'inventors' as domain, count(*) as count
                FROM applications a
                group by name
                having count(*) > 20 -- individuals unlikely to have more patents
            """,
            # ctgov db
            "aact": """
                select lower(name) as name, 'sponsors' as domain, count(*) as count
                from sponsors
                group by lower(name)
            """,
            # drugcentral db, with approvals
            # `ob_product`` has 1772 distinct applicants vs `approval` at 1136
            "drugcentral": """
                select lower(applicant) as name, 'applicants' as domain, count(*) as count
                from ob_product
                where applicant is not null
                group by lower(applicant)
            """,
        }
        rows = flatten(
            [
                PsqlDatabaseClient(db).select(query)
                for db, query in db_owner_query_map.items()
            ]
        )
        owners = clean_owners([row["name"] for row in rows])

        normalized: list[TermRecord] = [
            {
                "term": owner,
                "count": row["count"] or 0,
                "domain": row["domain"],
                "id": None,
                "ids": [],
                "original_term": row["name"],
            }
            for row, owner in zip(rows, owners)
            if len(owner) > 0
        ]

        terms = TermAssembler.__group_terms(normalized)
        return terms

    def generate_entity_terms(self) -> list[AggregatedTermRecord]:
        """
        Creates entity terms from the annotations table
        Normalizes terms and associates to canonical ids
        """
        terms_query = f"""
                SELECT lower(original_term) as term, domain, COUNT(*) as count
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
                where length(original_term) > 0
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
                "id": getattr(normalization_map.get(row["term"]) or (), "id", None),
                "ids": getattr(normalization_map.get(row["term"]) or (), "ids", None),
                "domain": row["domain"],
                "original_term": row["term"],
            }
            for row in rows
        ]

        return TermAssembler.__group_terms(term_records)

    def generate_terms(self) -> list[AggregatedTermRecord]:
        """
        Collects and forms terms for the terms table; persists to TERMS_FILE
        From
        - biosym_annotations
        - applications (assignees + inventors)
        """
        logging.info("Getting owner (assignee, inventor) terms")
        assignee_terms = self.generate_owner_terms()

        logging.info("Generating entity terms")
        entity_terms = self.generate_entity_terms()

        terms = assignee_terms + entity_terms

        # persisting to json (easy to replay if it borks on db ingest)
        save_json_as_file(terms, TERMS_FILE)
        return terms

    def persist_terms(self):
        """
        Persists terms (TERMS_FILE) to a table
        """
        terms = load_json_from_file(TERMS_FILE)

        schema = {
            "term": "TEXT",
            "id": "TEXT",
            "ids": "TEXT[]",
            "count": "INTEGER",
            "domains": "TEXT[]",
            "synonyms": "TEXT[]",
            "text_search": "tsvector",
        }
        self.client.create_and_insert(
            terms, TERMS_TABLE, schema, truncate_if_exists=True
        )
        logging.info(f"Inserted %s rows into terms table", len(terms))

    def index_terms(self):
        """
        Create search column / index on terms table
        """
        self.client.execute_query(
            f"""
            UPDATE {TERMS_TABLE}
            SET text_search = to_tsvector('english', ARRAY_TO_STRING(synonyms, '|| " " ||'));
            """
        )

        self.client.create_indices(
            [
                {
                    "table": TERMS_TABLE,
                    "column": "id",
                },
                {
                    "table": TERMS_TABLE,
                    "column": "ids",
                    "is_gin": True,
                    "is_lower": False,
                },
                {
                    "table": TERMS_TABLE,
                    "column": "text_search",
                    "is_gin": True,
                    "is_lower": False,
                },
            ]
        )

    def generate_and_persist_terms(self):
        """
        Generate and persist terms
        """
        self.generate_terms()
        self.persist_terms()
        self.index_terms()

    @staticmethod
    def run():
        """
        Create the terms and synonym map tables

        Idempotent (all tables are dropped and recreated)
        """
        TermAssembler().generate_and_persist_terms()
        sm = SynonymMapper(SYNONYM_MAP)
        sm.add_synonyms()
        sm.index()


def create_patent_terms():
    """
    Create the terms and synonym map tables

    Idempotent (all tables are dropped and recreated)

    $ jq '.[] | select(.name == "therapeutically")' terms.json
    """
    # populate_working_biosym_annotations()

    TermAssembler.run()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.terms")
        sys.exit()

    create_patent_terms()
