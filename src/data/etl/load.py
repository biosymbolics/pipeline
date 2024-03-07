import asyncio
import sys

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import SEARCH_TABLE
from typings.documents.common import DocType, VectorizableRecordType
from .entity import BiomedicalEntityLoader, OwnerLoader, UmlsLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


def get_entity_map_matview_query() -> list[str]:
    """
    Create query for search matterialized view

    TODO: Move
    """
    name_fields = [
        "e.name",
        "canonical_name",
        "instance_rollup",
        "category_rollup",
    ]
    search_or = "|| ' ' || ".join(name_fields)
    search = f"to_tsvector('english', {search_or} )"

    doc_type_switch = " ".join(
        [
            f"WHEN {doc_type.name}_id IS NOT NULL THEN '{doc_type.name}'"
            for doc_type in DocType
            if doc_type != DocType.all
        ]
    )

    fields = [
        *[f"{doc_type.name}_id" for doc_type in DocType if doc_type != DocType.all],
        f"{search} as search",
        *name_fields,
        f"CASE {doc_type_switch} END as doc_type",
    ]
    return [
        f"DROP MATERIALIZED VIEW IF EXISTS {SEARCH_TABLE}",
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {SEARCH_TABLE} AS
            (
                SELECT {', '.join(fields)}, canonical_type as type, entity_id
                FROM intervenable e

                UNION ALL

                SELECT {', '.join(fields)}, canonical_type as type, entity_id
                FROM indicatable e

                UNION ALL

                SELECT {', '.join(fields)}, 'OWNER' as type, owner_id as entity_id
                FROM ownable e, owner where owner.id=e.owner_id AND owner_type<>'OTHER'
            )
        """,
        f"CREATE INDEX {SEARCH_TABLE}_search ON {SEARCH_TABLE} USING GIN(search)",
        *[
            f"CREATE INDEX {SEARCH_TABLE}_{doc_type.name}_id ON {SEARCH_TABLE} ({doc_type.name}_id)"
            for doc_type in DocType
            if doc_type != DocType.all
        ],
        f"CREATE INDEX {SEARCH_TABLE}_type ON {SEARCH_TABLE} (type)",
        f"CREATE INDEX {SEARCH_TABLE}_doc_type ON {SEARCH_TABLE} (doc_type)",
    ]


# these get wiped for every prisma db push. Will figure out a better way to handle this.
# https://github.com/prisma/prisma/issues/12751
MANUAL_INDICES = [
    "SET maintenance_work_mem = '5GB'",
    """
    CREATE INDEX IF NOT EXISTS patent_vector ON patent USING hnsw (vector vector_cosine_ops);
    """,
    """
    CREATE INDEX IF NOT EXISTS trial_vector ON trial USING hnsw (vector vector_cosine_ops);
    """,
    """
    CREATE INDEX IF NOT EXISTS umls_vector ON umls USING hnsw (vector vector_cosine_ops);
    """,
    """
    CREATE INDEX IF NOT EXISTS regulatory_approval_vector ON regulatory_approval USING hnsw (vector vector_cosine_ops);
    """
    """,
    CREATE INDEX owner_vector ON owner USING hnsw (vector vector_cosine_ops);
    """,
    """
    CREATE INDEX biomedical_entity_search ON biomedical_entity USING GIN(search);
    """,
]


async def load_all(force_update: bool = False):
    """
    Central script for stage 2 of ETL (local dbs -> biosym)

    Args:
        force_update (bool, optional): Whether to update or merely create.
            If update, documents and their relations are first deleted.
    """
    # copy umls data
    await UmlsLoader(
        record_type=VectorizableRecordType.umls, source_db="umls"
    ).copy_all()

    # copy all biomedical entities (from all doc types)
    # Takes 3+ hours!!
    await BiomedicalEntityLoader().copy_all()

    # copy owner data (across all documents)
    await OwnerLoader().copy_all(force_update)

    # copy patent data
    await PatentLoader(document_type=DocType.patent, source_db="patents").copy_all(
        force_update
    )

    # copy data about approvals
    await RegulatoryApprovalLoader(
        document_type=DocType.regulatory_approval, source_db="drugcentral"
    ).copy_all(force_update)

    # copy trial data
    await TrialLoader(document_type=DocType.trial, source_db="aact").copy_all(
        force_update
    )

    # do final biomedical entity stuff that requires everything else be in place
    await BiomedicalEntityLoader().finalize()

    # finally, link owners
    await OwnerLoader().finalize()

    # create some materialized views for reporting
    for query in get_entity_map_matview_query():
        await PsqlDatabaseClient("biosym").execute_query(query)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.load [--force_update]
            UMLS etl
        """
        )
        sys.exit()

    force_update = "--force_update" in sys.argv

    asyncio.run(load_all(force_update))
