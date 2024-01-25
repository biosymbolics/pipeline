"""
Utils for ETLing UMLs data
"""
import asyncio
import sys
import logging
from prisma.models import UmlsGraph, Umls
from prisma.types import UmlsGraphCreateWithoutRelationsInput as UmlsGraphRecord

from system import initialize

initialize()

from clients.low_level.prisma import batch_update, prisma_client
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL
from constants.umls import BIOMEDICAL_GRAPH_UMLS_TYPES

from .transform import UmlsAncestorTransformer, UmlsTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SOURCE_DB = "umls"
BATCH_SIZE = 10000


class UmlsLoader:
    @staticmethod
    async def create_umls_lookup():
        """
        Create UMLS lookup table

        Creates a table of UMLS entities: id, name, ancestor ids
        """

        source_sql = f"""
            SELECT
                TRIM(entities.cui) as id,
                TRIM(max(entities.str)) as name,
                COALESCE(TRIM(max(ancestors.ptr)), "") as hierarchy,
                COALESCE(array_agg(distinct semantic_types.tui::text), []::text) as type_ids,
                COALESCE(array_agg(distinct semantic_types.sty::text), []::text) as type_names,
                COALESCE(max(descendants.count), 0) as num_descendants,
                COALESCE(max(synonyms.terms), []::text) as synonyms
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
            AND entities.ts='P' -- preferred terms (according to UMLS)
            AND entities.ispref='Y' -- preferred term (according to UMLS)
            AND entities.stt='PF' -- preferred term (according to UMLS)
            GROUP BY entities.cui -- because multiple preferred terms (according to UMLS)
        """

        umls_db = f"{BASE_DATABASE_URL}/umls"
        aui_lookup = {
            r["aui"]: r["cui"]
            for r in await PsqlDatabaseClient(umls_db).select(
                "select TRIM(aui) aui, TRIM(cui) cui from mrconso"
            )
        }

        transform = UmlsTransformer(aui_lookup)

        client = await prisma_client(600)

        async def handle_batch(batch: list[dict]):
            logger.info("Creating %s UMLS records", len(batch))
            insert_data = [transform(r) for r in batch]
            await Umls.prisma(client).create_many(
                data=insert_data,
                skip_duplicates=True,
            )

        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=source_sql, batch_size=BATCH_SIZE, handle_result_batch=handle_batch
        )

    @staticmethod
    async def update_with_ontology_level():
        """
        Adds ontology levels (heavy, with dependencies)

        Run *after* BiomedicalEntityEtl
        """
        client = await prisma_client(600)
        all_umls = await Umls.prisma(client).find_many()

        # might be slow, if doing betweenness centrality calc.
        ult = await UmlsAncestorTransformer.create(all_umls)

        await batch_update(
            all_umls,
            update_func=lambda r, tx: Umls.prisma(tx).update(
                data=ult.transform(r), where={"id": r.id}
            ),
            batch_size=1000,  # gets mad if more than 1000 or so
        )

    @staticmethod
    async def copy_relationships():
        """
        Copy relationships from umls to patents

        - Creates a table of UMLS relationships: head_id, head_name, tail_id, tail_name, relationship
        - Limits based on language and semantic type
        """
        source_sql = f"""
            SELECT
                cui1 as head_id, max(head.str) as head_name,
                cui2 as tail_id, max(tail.str) as tail_name,
                COALESCE(rela, '') as relationship
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

        async def handle_batch(batch):
            logger.info("Creating %s UmlsGraph records", len(batch))
            await UmlsGraph.prisma().create_many(
                data=[UmlsGraphRecord(**r) for r in batch],
                skip_duplicates=True,
            )

        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=source_sql, batch_size=BATCH_SIZE, handle_result_batch=handle_batch
        )

    @staticmethod
    async def copy_all():
        """
        Copy all UMLS data
        """
        await UmlsLoader.create_umls_lookup()
        await UmlsLoader.copy_relationships()

    @staticmethod
    async def update_all():
        await UmlsLoader.update_with_ontology_level()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.biomedical_entity.umls.load
            UMLS etl
        """
        )
        sys.exit()

    asyncio.run(UmlsLoader.copy_all())
    # asyncio.run(UmlsLoader.update_all())
