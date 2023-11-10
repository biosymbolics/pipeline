"""
Utils for ETLing UMLs data
"""
import sys
import logging
from typing import Any, Sequence, TypeGuard

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from typings.umls import OntologyLevel, UmlsRecord

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


def transform_umls_relationships(
    records: Sequence[dict], aui_lookup: dict[str, str]
) -> list[dict[str, Any]]:
    """
    Transform umls relationship
    """
    if not is_umls_record_list(records):
        raise ValueError(f"Records are not UmlsRecords: {records[:10]}")

    def parse_ancestor_ids(record: UmlsRecord) -> dict[str, Any]:
        hier = record["hierarchy"]
        # reverse to get nearest ancestor first
        ancestors = (hier.split(".") if hier is not None else [])[::-1]

        # TODO strip shouldn't be necessary
        ancestor_cuis = [aui_lookup.get(aui, "").strip() for aui in ancestors]

        return {
            **record,  # type: ignore
            "level": OntologyLevel.find(record, ancestor_cuis),
            **{f"l{i}_ancestor": None for i in range(MAX_DENORMALIZED_ANCESTORS)},
            **{
                f"l{i}_ancestor": ancestor_cuis[i] if i < len(ancestors) else None
                for i in range(MAX_DENORMALIZED_ANCESTORS)
            },
        }

    return [parse_ancestor_ids(r) for r in records]


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
            entities.cui as id,
            max(entities.str) as canonical_name,
            max(ancestors.ptr) as hierarchy,
            {", ".join(ANCESTOR_FIELDS)},
            max(semantic_types.tui) as type_id,
            max(semantic_types.sty) as type_name,
            '' as level,
            max(descendants.count) as num_descendants
        from mrconso as entities
        LEFT JOIN mrhier as ancestors on ancestors.cui = entities.cui
        JOIN (
            select cui1 as parent_cui, count(*) as count
            from mrrel
            where rel = 'RN' -- narrower
            and rela is null -- no specified relationship
            group by parent_cui
        ) descendants ON descendants.parent_cui = entities.cui
        JOIN mrsty as semantic_types on semantic_types.cui = entities.cui
        where entities.lat='ENG' -- english
        AND entities.ts='P' -- preferred terms
        AND entities.ispref='Y' -- preferred term
        group by entities.cui -- because multiple preferred terms
    """

    new_table_name = "umls_lookup"

    umls_db = f"{BASE_DATABASE_URL}/umls"
    aui_lookup = {
        r["aui"]: r["cui"]
        for r in PsqlDatabaseClient(umls_db).select("select aui, cui from mrconso")
    }

    PsqlDatabaseClient(umls_db).truncate_table(new_table_name)
    PsqlDatabaseClient.copy_between_db(
        umls_db,
        source_sql,
        f"{BASE_DATABASE_URL}/patents",
        new_table_name,
        transform=lambda records: transform_umls_relationships(records, aui_lookup),
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
