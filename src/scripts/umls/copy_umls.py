"""
Utils for ETLing UMLs data
"""
import sys
import logging
from typing import Sequence, TypeGuard, cast

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from data.common.biomedical.umls import clean_umls_name
from typings.umls import OntologyLevel, UmlsRecord, UmlsLookupRecord

from .ancestor_selection import UmlsGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MAX_DENORMALIZED_ANCESTORS = 10


def is_umls_record_list(
    records: Sequence[dict],
) -> TypeGuard[Sequence[UmlsRecord]]:
    """
    Check if list of records is a list of UmlsRecords
    """
    return (
        isinstance(records, list)
        and len(records) > 0
        and isinstance(records[0], dict)
        and "id" in records[0]
        and "canonical_name" in records[0]
        and "hierarchy" in records[0]
        and "type_id" in records[0]
        and "type_name" in records[0]
        and "level" in records[0]
    )


class UmlsTransformer:
    def __init__(self, aui_lookup: dict[str, str]):
        self.aui_lookup: dict[str, str] = aui_lookup
        self.lookup_dict: dict[str, UmlsLookupRecord] | None = None
        self.betweenness_map: dict[str, float] = UmlsGraph().betweenness_map

    def initialize(self, all_records: Sequence[dict]):
        if not is_umls_record_list(all_records):
            raise ValueError(f"Records are not UmlsRecords: {all_records[:10]}")

        logger.info("Initializing UMLS lookup dict with %s records", len(all_records))
        lookup_records = [self.create_lookup_record(r) for r in all_records]
        self.lookup_dict = {r["id"]: r for r in lookup_records}
        logger.info("Initializing UMLS lookup dict")

    def create_lookup_record(self, record: UmlsRecord) -> UmlsLookupRecord:
        hier = record["hierarchy"]
        # reverse to get nearest ancestor first
        ancestors = (hier.split(".") if hier is not None else [])[::-1]
        ancestor_cuis = [self.aui_lookup.get(aui, "") for aui in ancestors]

        return {
            **record,  # type: ignore
            **{f"l{i}_ancestor": None for i in range(MAX_DENORMALIZED_ANCESTORS)},
            **{
                f"l{i}_ancestor": ancestor_cuis[i] if i < len(ancestor_cuis) else None
                for i in range(MAX_DENORMALIZED_ANCESTORS)
            },
            "level": OntologyLevel.find(record["id"], self.betweenness_map),
            "preferred_name": clean_umls_name(
                record["id"], record["canonical_name"], record["synonyms"]
            ),
        }

    def find_level_ancestor(
        self,
        record: UmlsLookupRecord,
        level: OntologyLevel,
    ) -> str:
        """
        Find first ancestor at the specified level

        Args:
            record (UmlsLookupRecord): UMLS record
            level (OntologyLevel): level to find

        Returns (str): ancestor id, or "" if none found
        """
        if self.lookup_dict is None:
            raise ValueError("Lookup dict is not initialized")

        # use self as rollup if at the right level
        if record["level"] == level:
            return record["id"]

        # take the first ancestor that is at the right level
        for i in range(MAX_DENORMALIZED_ANCESTORS):
            acui = record[f"l{i}_ancestor"]
            if acui is not None and acui in self.lookup_dict:
                ancestor_rec = self.lookup_dict[acui]
                if ancestor_rec["level"] == level:
                    return ancestor_rec["id"]
            elif acui is not None:
                logger.error("Missing ancestor %s for %s", acui, record["id"])

        return ""

    def __call__(
        self,
        batch: Sequence[dict],
        all_records: Sequence[dict],
    ) -> list[UmlsLookupRecord]:
        """
        Transform umls relationship

        Args:
            batch (Sequence[dict]): batch of records to transform
            all_records (Sequence[dict]): all records
        """
        if not is_umls_record_list(batch):
            raise ValueError(f"Records are not UmlsRecords: {batch[:10]}")

        if self.lookup_dict is None:
            self.initialize(all_records)
            assert self.lookup_dict is not None

        batch_records = [self.lookup_dict[r["id"]] for r in batch]

        return [
            cast(
                UmlsLookupRecord,
                {
                    **r,  # type: ignore
                    "instance_rollup": self.find_level_ancestor(
                        r, OntologyLevel.L1_CATEGORY
                    ),
                    "category_rollup": self.find_level_ancestor(
                        r, OntologyLevel.L2_CATEGORY
                    ),
                },
            )
            for r in batch_records
        ]


def create_umls_lookup():
    """
    Create UMLS lookup table

    - Creates a table of UMLS entities: id, name, ancestor ids
    """

    ANCESTOR_FIELDS = [
        f"'' as l{i}_ancestor" for i in range(MAX_DENORMALIZED_ANCESTORS)
    ]

    source_sql = f"""
        select
            TRIM(entities.cui) as id,
            TRIM(max(entities.str)) as canonical_name,
            TRIM(max(ancestors.ptr)) as hierarchy,
            {", ".join(ANCESTOR_FIELDS)},
            '' as preferred_name,
            '' as level,
            '' as instance_rollup,
            '' as category_rollup,
            TRIM(max(semantic_types.tui)) as type_id,
            TRIM(max(semantic_types.sty)) as type_name,
            COALESCE(max(descendants.count), 0) as num_descendants,
            max(synonyms.terms) as synonyms
        from mrconso as entities
        LEFT JOIN mrhier as ancestors on ancestors.cui = entities.cui
        LEFT JOIN (
            select cui1 as parent_cui, count(*) as count
            from mrrel
            where rel in ('RN', 'CHD') -- narrower, child
            and (rela is null or rela = 'isa') -- no specified relationship, or 'is a' rel
            group by parent_cui
        ) descendants ON descendants.parent_cui = entities.cui
        LEFT JOIN mrsty as semantic_types on semantic_types.cui = entities.cui
        LEFT JOIN (
            select array_agg(distinct(lower(str))) as terms, cui as id from mrconso
            group by cui
        ) as synonyms on synonyms.id = entities.cui
        where entities.lat='ENG' -- english
        AND entities.ts='P' -- preferred terms
        AND entities.ispref='Y' -- preferred term
        group by entities.cui -- because multiple preferred terms
    """

    new_table_name = "umls_lookup"

    umls_db = f"{BASE_DATABASE_URL}/umls"
    aui_lookup = {
        r["aui"]: r["cui"]
        for r in PsqlDatabaseClient(umls_db).select(
            "select TRIM(aui) aui, TRIM(cui) cui from mrconso"
        )
    }

    PsqlDatabaseClient(umls_db).truncate_table(new_table_name)
    transform = UmlsTransformer(aui_lookup)
    PsqlDatabaseClient.copy_between_db(
        umls_db,
        source_sql,
        f"{BASE_DATABASE_URL}/patents",
        new_table_name,
        transform=lambda batch, all_records: transform(batch, all_records),
    )

    PsqlDatabaseClient().create_indices(
        [
            {
                "table": new_table_name,
                "column": "id",
            },
            *[
                {"table": new_table_name, "column": f"l{i}_ancestor"}
                for i in range(MAX_DENORMALIZED_ANCESTORS)
            ],
        ]
    )


def copy_relationships():
    """
    Copy relationships from umls to patents
    """
    # cui1 inverse_isa cui2 for parent (cui2 is parent); rel=PAR
    source_sql = """
        select
        cui1 as head_id, max(head.str) as head_name,
        cui2 as tail_id, max(tail.str) as tail_name,
        rela as rel_type
        from mrrel, mrconso head, mrconso tail
        where head.cui = mrrel.cui1
        AND tail.cui = mrrel.cui2
        AND head.lat='ENG'
        AND head.ts='P'
        AND head.ispref='Y'
        AND tail.lat='ENG'
        AND tail.ts='P'
        AND tail.ispref='Y'
        group by head_id, tail_id, rel_type
    """

    new_table_name = "umls_graph"

    umls_db = f"{BASE_DATABASE_URL}/umls"
    PsqlDatabaseClient(umls_db).truncate_table(new_table_name)
    PsqlDatabaseClient.copy_between_db(
        umls_db, source_sql, f"{BASE_DATABASE_URL}/patents", new_table_name
    )

    PsqlDatabaseClient().create_indices(
        [
            {
                "table": new_table_name,
                "column": "head_id",
            },
            {
                "table": new_table_name,
                "column": "rel_type",
            },
        ]
    )


def copy_umls():
    """
    Copy data from umls to patents
    """
    create_umls_lookup()
    # copy_relationships()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.umls.copy_umls
            Copies umls to patents
        """
        )
        sys.exit()

    copy_umls()
