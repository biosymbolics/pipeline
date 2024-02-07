import asyncio
import sys

from clients.low_level.postgres import PsqlDatabaseClient
from typings.documents.common import DocType
from .entity import BiomedicalEntityLoader, OwnerLoader, UmlsLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


def get_entity_map_matview_query(doc_type: DocType) -> list[str]:
    """
    Create query for search matterialized view
    """
    name_fields = [
        "name",
        "canonical_name",
        "instance_rollup",
        "category_rollup",
    ]
    search_or = "|| ' ' || ".join(name_fields)
    search = f"to_tsvector('english', {search_or} )"
    fields = [
        f"{doc_type.name}_id",
        f"{search} as search",
        *name_fields,
    ]
    return [
        f"""
        CREATE MATERIALIZED VIEW IF NOT EXISTS {doc_type.name}_entity_map AS
            (
                SELECT {', '.join(fields)}, canonical_type as type
                FROM intervenable
                WHERE {doc_type.name}_id is not null

                UNION ALL

                SELECT {', '.join(fields)}, canonical_type as type
                FROM indicatable
                WHERE {doc_type.name}_id is not null

                UNION ALL

                SELECT {', '.join(fields)}, 'OWNER' as type
                FROM ownable
                WHERE {doc_type.name}_id is not null
            )
        """,
        f"CREATE INDEX idx_{doc_type.name}_entity_map ON {doc_type.name}_entity_map USING GIN(search)",
    ]


async def load_all(force_update: bool = False):
    """
    Central script for stage 2 of ETL (local dbs -> biosym)

    Args:
        force_update (bool, optional): Whether to update or merely create.
            If update, documents and their relations are first deleted.
    """
    # copy umls data
    # await UmlsLoader().copy_all()

    # # copy all biomedical entities (from all doc types)
    # # Takes 3+ hours!!
    # await BiomedicalEntityLoader().copy_all()

    # # copy owner data (across all documents)
    # await OwnerLoader().copy_all()

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
    for doc_type in DocType:
        for query in get_entity_map_matview_query(doc_type):
            await PsqlDatabaseClient("biosym").execute_query(query)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.load
            UMLS etl
        """
        )
        sys.exit()

    force_update = "--force_update" in sys.argv

    asyncio.run(load_all(force_update))
