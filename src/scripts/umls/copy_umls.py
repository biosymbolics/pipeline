"""
Utils for ETLing UMLs data
"""
import sys
import logging

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from constants.umls import BIOMEDICAL_GRAPH_UMLS_TYPES

from .constants import MAX_DENORMALIZED_ANCESTORS
from .transform import UmlsTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UMLS_LOOKUP_TABLE = "umls_lookup"


def create_umls_lookup():
    """
    Create UMLS lookup table

    - Creates a table of UMLS entities: id, name, ancestor ids
    """

    ANCESTOR_FIELDS = [
        f"'' as l{i}_ancestor" for i in range(MAX_DENORMALIZED_ANCESTORS)
    ]

    source_sql = f"""
        SELECT
            TRIM(entities.cui) as id,
            TRIM(max(entities.str)) as canonical_name,
            TRIM(max(ancestors.ptr)) as hierarchy,
            array_agg(distinct semantic_types.tui::text) as type_ids,
            array_agg(distinct semantic_types.sty::text) as type_names,
            COALESCE(max(descendants.count), 0) as num_descendants,
            max(synonyms.terms) as synonyms,
            {", ".join(ANCESTOR_FIELDS)},
            '' as preferred_name,
            '' as level,
            '' as instance_rollup,
            '' as category_rollup
        FROM mrconso as entities
        LEFT JOIN mrhier as ancestors on ancestors.cui = entities.cui
        LEFT JOIN (
            SELECT cui1 as parent_cui, count(*) as count
            FROM mrrel
            WHERE rel in ('RN', 'CHD') -- narrower, child
            AND (rela is null or rela = 'isa') -- no specified relationship, or 'is a' rel
            GROUP BY parent_cui
        ) descendants ON descendants.parent_cui = entities.cui
        LEFT JOIN mrsty as semantic_types on semantic_types.cui = entities.cui
        LEFT JOIN (
            select array_agg(distinct(lower(str))) as terms, cui as id from mrconso
            group by cui
        ) as synonyms on synonyms.id = entities.cui
        WHERE entities.lat='ENG' -- english
        AND entities.ts='P' -- preferred terms
        AND entities.ispref='Y' -- preferred term
        GROUP BY entities.cui -- because multiple preferred terms
    """

    umls_db = f"{BASE_DATABASE_URL}/umls"
    aui_lookup = {
        r["aui"]: r["cui"]
        for r in PsqlDatabaseClient(umls_db).select(
            "select TRIM(aui) aui, TRIM(cui) cui from mrconso"
        )
    }

    PsqlDatabaseClient(umls_db).truncate_table(UMLS_LOOKUP_TABLE)
    transform = UmlsTransformer(aui_lookup)
    PsqlDatabaseClient.copy_between_db(
        umls_db,
        source_sql,
        f"{BASE_DATABASE_URL}/patents",
        UMLS_LOOKUP_TABLE,
        transform=lambda batch, all_records: transform(batch, all_records),
    )

    PsqlDatabaseClient().create_indices(
        [
            {
                "table": UMLS_LOOKUP_TABLE,
                "column": "id",
            },
            {
                "table": UMLS_LOOKUP_TABLE,
                "column": "type_ids",
                "is_gin": True,
            },
            {
                "table": UMLS_LOOKUP_TABLE,
                "column": "canonical_name",
                "is_tgrm": True,
            },
        ]
    )


def copy_relationships():
    """
    Copy relationships from umls to patents

    - Creates a table of UMLS relationships: head_id, head_name, tail_id, tail_name, relationship
    - Limits based on language and semantic type
    """
    source_sql = f"""
        SELECT
            cui1 as head_id, max(head.str) as head_name,
            cui2 as tail_id, max(tail.str) as tail_name,
            rela as relationship
        FROM mrrel
        JOIN mrconso as head on head.cui = cui1
        JOIN mrconso as tail on tail.cui = cui2
        JOIN mrsty as head_semantic_type on head_semantic_type.cui = cui1
        JOIN mrsty as tail_semantic_type on tail_semantic_type.cui = cui2
        WHERE head.lat='ENG'
        AND head.ts='P'
        AND head.ispref='Y'
        AND tail.lat='ENG'
        AND tail.ts='P'
        AND tail.ispref='Y'
        AND head_semantic_type.tui in {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
        AND tail_semantic_type.tui in {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
        GROUP BY head_id, tail_id, relationship
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
                "column": "relationship",
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
