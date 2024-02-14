import asyncio
import sys

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import SEARCH_TABLE
from typings.documents.common import DocType
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
    fields = [
        *[f"{doc_type.name}_id" for doc_type in DocType],
        f"{search} as search",
        *name_fields,
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
        f"CREATE INDEX IF NOT EXISTS idx_{SEARCH_TABLE}_search ON {SEARCH_TABLE} USING GIN(search)",
        *[
            f"CREATE INDEX IF NOT EXISTS idx_{SEARCH_TABLE}_{doc_type.name}_id ON {SEARCH_TABLE} ({doc_type.name}_id)"
            for doc_type in DocType
        ],
        f"CREATE INDEX IF NOT EXISTS idx_{SEARCH_TABLE}_type ON {SEARCH_TABLE} (type)",
    ]


async def load_all(force_update: bool = False):
    """
    Central script for stage 2 of ETL (local dbs -> biosym)

    Args:
        force_update (bool, optional): Whether to update or merely create.
            If update, documents and their relations are first deleted.
    """
    # # copy umls data
    # await UmlsLoader().copy_all()

    # # copy all biomedical entities (from all doc types)
    # # Takes 3+ hours!!
    # await BiomedicalEntityLoader().copy_all()

    # # copy owner data (across all documents)
    # await OwnerLoader().copy_all(force_update)

    # # copy patent data
    # await PatentLoader(document_type="patent").copy_all(force_update)

    # # copy data about approvals
    # await RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all(
    #     force_update
    # )

    # # copy trial data
    # await TrialLoader(document_type="trial").copy_all(force_update)

    # # do final biomedical entity stuff that requires everything else be in place
    # await BiomedicalEntityLoader().post_finalize()

    # # finally, link owners
    # await OwnerLoader().post_finalize()

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
