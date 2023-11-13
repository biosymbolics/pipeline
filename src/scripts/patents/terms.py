"""
Functions to initialize the terms and synonym tables
"""
import sys
from typing import Sequence, TypeGuard, TypedDict
import logging
from pydash import compact, flatten, group_by, uniq

import system

system.initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import WORKING_BIOSYM_ANNOTATIONS_TABLE
from core.ner import TermNormalizer
from core.ner.utils import lemmatize_tails
from utils.file import load_json_from_file, save_json_as_file
from utils.list import dedup

from .constants import TERMS_FILE, TERMS_TABLE
from .process_biosym_annotations import populate_working_biosym_annotations
from .synonyms import SynonymMapper
from .types import AggregatedTermRecord, Ancestors, TermRecord
from .utils import clean_owners


MIN_CANONICAL_NAME_COUNT = 4

CanonicalRecord = TypedDict(
    "CanonicalRecord",
    {
        "id": str,
        "canonical_name": str,
        "preferred_name": str,
        "instance_rollup": str,
        "category_rollup": str,
    },
)
CanonicalMap = dict[str, CanonicalRecord]


def is_canonical_record(record: dict) -> TypeGuard[CanonicalRecord]:
    """
    Check if record is a CanonicalRecord
    """
    return (
        isinstance(record, dict)
        and "id" in record
        and "canonical_name" in record
        and "instance_rollup" in record
        and "category_rollup" in record
    )


def is_canonical_records(
    records: Sequence[dict],
) -> TypeGuard[Sequence[CanonicalRecord]]:
    return all([is_canonical_record(r) for r in records])


class TermAssembler:
    """
    Static class for assembling terms and synonyms

    Terms assembled:
    - owner/sponsor/applications (companies, universities and individuals)
    - entities/annotations (compounds, diseases, mechanisms)
    """

    def __init__(self):
        self.client = PsqlDatabaseClient()
        self.canonical_map = self._load_canonical()

    def _load_canonical(self) -> CanonicalMap:
        """
        Load some UMLs data for canonical names
        """
        canonical_records = self.client.select(
            "select id, canonical_name, preferred_name, instance_rollup, category_rollup from umls_lookup"
        )
        if not is_canonical_records(canonical_records):
            raise ValueError(
                f"Records are not CanonicalRecords: {canonical_records[:10]}"
            )
        return {row["id"]: row for row in canonical_records}

    @staticmethod
    def _get_preferred_name(term_records: Sequence[TermRecord]) -> str:
        """
        Get preferred name from a list of term records with the same id

        Uses the most common "original_term" if used with sufficiently high frequency,
        and otherwise uses the "term" (often a mangled composite of UMLS terms, which is why we don't use it by default)
        """
        if len(uniq([f"{row['id']}{row['term']}" for row in term_records])) > 1:
            raise ValueError("Term records must have the same id and name")

        canonical_term = term_records[0]["term"]

        # use canonical name if there is only one id
        if len(term_records[0]["ids"] or []) == 1:
            return canonical_term

        # otherwise, group the records so we can determine which term has the most synonyms
        grouped_synonyms = sorted(
            group_by(term_records, "original_term").items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        # if the top term is sufficiently common, use that as the preferred name
        if len(grouped_synonyms[0][1]) > MIN_CANONICAL_NAME_COUNT:
            return grouped_synonyms[0][0]

        # hydrolytic Enzyme Inhibitors
        # otherwise, use the canonical term
        return canonical_term

    @staticmethod
    def _group_terms(terms: list[TermRecord]) -> list[AggregatedTermRecord]:
        """
        Dedups/groups terms by id and term
        """
        # depends upon id always being in the same order (enforced elsewhere)
        grouped_terms: dict[str, Sequence[TermRecord]] = group_by(
            [{**t, "key": t["id"] or t["term"]} for t in terms],  # type: ignore
            "key",
        )

        def __get_term_record(group: Sequence[TermRecord]) -> AggregatedTermRecord:
            preferred_name = TermAssembler._get_preferred_name(group)
            return {
                "term": preferred_name,
                "count": sum(row["count"] for row in group),
                "id": group[0].get("id") or "",
                "ids": group[0].get("ids") or [],
                "domains": dedup([row["domain"] for row in group]),
                # lemmatize tails for less duplication. todo: lemmatize all?
                # 2x dedup for perf
                # TODO: add synonyms from UMLS
                "synonyms": dedup(
                    lemmatize_tails(
                        dedup([row["original_term"] or "" for row in group])
                    )
                ),
            }

        agg_terms = [__get_term_record(group) for _, group in grouped_terms.items()]
        return agg_terms

    def _generate_owner_terms(self) -> list[AggregatedTermRecord]:
        """
        Generates owner terms (assignee/inventor) from:
        - patent applications table
        - aact (ctgov)
        - drugcentral approvals
        """
        # uses this to roughly select for companies & universities over individuals
        # otherwise the imperfect grouping gets confused
        ASSIGNEE_PATENT_THRESHOLD = 20
        db_owner_query_map = {
            # patents db
            "patents": f"""
                SELECT lower(unnest(assignees)) as name, 'assignees' as domain, count(*) as count
                FROM applications a
                group by name
                having count(*) > {ASSIGNEE_PATENT_THRESHOLD} -- individuals unlikely to have more patents

                UNION ALL

                SELECT lower(unnest(inventors)) as name, 'inventors' as domain, count(*) as count
                FROM applications a
                group by name
                having count(*) > {ASSIGNEE_PATENT_THRESHOLD}
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

        terms = TermAssembler._group_terms(normalized)
        return terms

    def _generate_entity_terms(self) -> list[AggregatedTermRecord]:
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
        norm_map = dict(normalizer.normalize(terms))
        logging.info("Finished creating normalization_map")

        def __normalize(term: str, domain: str) -> str:
            if domain == "attributes":
                return term  # leave attributes alone
            entry = norm_map.get(term)
            if not entry:
                return term
            return entry.name

        term_records: list[TermRecord] = [
            {
                "term": __normalize(row["term"], row["domain"]),
                "count": row["count"] or 0,
                "id": getattr(norm_map.get(row["term"]) or (), "id", None),
                "ids": getattr(norm_map.get(row["term"]) or (), "ids", None),
                "domain": row["domain"],
                "original_term": row["term"],
            }
            for row in rows
        ]

        return TermAssembler._group_terms(term_records)

    def generate_terms(self) -> list[AggregatedTermRecord]:
        """
        Collects and forms terms for the terms table; persists to TERMS_FILE
        From
        - biosym_annotations
        - applications (assignees + inventors)
        """
        logging.info("Getting owner (assignee, inventor) terms")
        assignee_terms = self._generate_owner_terms()

        logging.info("Generating entity terms")
        entity_terms = self._generate_entity_terms()

        terms = assignee_terms + entity_terms

        # persisting to json (easy to replay if it borks on db ingest)
        save_json_as_file(terms, TERMS_FILE)
        return terms

    def get_ancestors(self, record: AggregatedTermRecord) -> Ancestors | dict:
        """
        Get ancestor names for record (used for aggregation)

        If no instance or category ancestor is found, it will return the term itself
        (to simplify the use of these fields as rollups)
        """

        def get_ancestor(ids: Sequence[str], type_name: str) -> str | None:
            ancestor_ids: list[str] = compact(
                [self.canonical_map.get(i, {}).get(f"{type_name}_rollup") for i in ids]
            )
            names = [
                self.canonical_map[ai]["preferred_name"]
                for ai in ancestor_ids
                if ai in self.canonical_map and ai != record["id"]
            ]
            if len(names) > 0:
                return names[0]
            return None

        ids = record["ids"] or []
        instance_rollup = get_ancestor(ids, "instance") or record["term"]
        category_rollup = get_ancestor(ids, "category") or instance_rollup

        return {
            "instance_rollup": instance_rollup,
            "category_rollup": category_rollup,
        }

    def persist_terms(self):
        """
        Persists terms (TERMS_FILE) to a table
        Applies a transformation (get_ancestors -> adding ancestor fields)
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
            "instance_rollup": "TEXT",
            "category_rollup": "TEXT",
        }
        self.client.create_and_insert(
            terms,
            TERMS_TABLE,
            schema,
            truncate_if_exists=True,
            transform=lambda batch, _: [{**r, **self.get_ancestors(r)} for r in batch],
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
        sm = SynonymMapper()
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
