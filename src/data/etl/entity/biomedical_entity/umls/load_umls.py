"""
Utils for ETLing UMLs data
"""

import asyncio
import sys
import logging
import time
from prisma.enums import OntologyLevel
from prisma.models import UmlsGraph, Umls
from prisma.types import (
    UmlsCreateWithoutRelationsInput as UmlsCreateInput,
    UmlsGraphCreateWithoutRelationsInput as UmlsGraphRecord,
)
from pydash import compact

from system import initialize

initialize()

from clients.low_level.prisma import batch_update, prisma_client
from clients.low_level.postgres import PsqlDatabaseClient
from constants.umls import BIOMEDICAL_GRAPH_UMLS_TYPES
from data.domain.biomedical.umls import clean_umls_name

from .umls_transform import UmlsAncestorTransformer

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
                COALESCE(array_agg(distinct semantic_types.tui::text), ARRAY[]::text[]) as type_ids,
                COALESCE(array_agg(distinct semantic_types.sty::text), ARRAY[]::text[]) as type_names,
                COALESCE(max(synonyms.terms), ARRAY[]::text[]) as synonyms
            FROM mrconso as entities
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

        client = await prisma_client(600)

        async def handle_batch(batch: list[dict]):
            logger.info("Creating %s UMLS records", len(batch))
            insert_data = [
                UmlsCreateInput(
                    id=r["id"],
                    name=r["name"],
                    rollup_id=r["id"],  # start with self as rollup
                    preferred_name=clean_umls_name(
                        r["id"], r["name"], r["synonyms"], r["type_ids"], False
                    ),
                    synonyms=r["synonyms"],
                    type_ids=r["type_ids"],
                    type_names=r["type_names"],
                    level=OntologyLevel.UNKNOWN,
                )
                for r in batch
            ]
            await Umls.prisma(client).create_many(
                data=insert_data,
                skip_duplicates=True,
            )

        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=source_sql, batch_size=BATCH_SIZE, handle_result_batch=handle_batch
        )

    @staticmethod
    async def set_ontology_levels():
        """
        Adds ontology levels

        Run *after* BiomedicalEntityEtl
        """
        start = time.monotonic()
        logger.info("Updating UMLS with levels")
        client = await prisma_client(600)

        # TODO: just grab nodes from graph
        all_umls = await Umls.prisma(client).find_many()

        # might be slow-ish, if creating a new graph
        ult = await UmlsAncestorTransformer.create()

        async def update(record, tx):
            if "id" not in record:
                raise ValueError(f"Record missing id: {record}")

            return await Umls.prisma(tx).update(data=record, where={"id": record["id"]})

        update_records = compact([ult.transform(r) for r in all_umls])
        await batch_update(
            update_records,
            update_func=update,
            batch_size=10000,
        )
        logger.info(
            "Finished updating UMLS with levels in %s seconds",
            round(time.monotonic() - start),
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
    async def add_missing_relationships():
        """
        Add missing UMLS relationships
        """
        client = await prisma_client(600)
        # if no 'isa' relationship exists for the tail
        # AND the tail name is a subset of the head synonyms (e.g. XYZ inhibitor -> XYZ)
        # then mark the relationship as 'isa'
        create_fun_query = """
            create or replace function word_ngrams(str text, n int)
            returns setof text language plpgsql as $$
            declare
                i int;
                arr text[];
            begin
                arr := regexp_split_to_array(str, '[^[:alnum:]]+');
                for i in 1 .. cardinality(arr)- n+ 1 loop
                    return next array_to_string(arr[i : i+n-1], ' ');
                end loop;
            end $$;
        """
        await client.execute_raw(create_fun_query)
        query = """
            update umls_graph ug set relationship='isa'
            FROM umls head_umls
            LEFT JOIN umls_graph ug2 ON ug2.tail_id=head_umls.id and ug2.relationship='isa'
            where ug.head_id=head_umls.id
            and ARRAY['T116', 'T028', 'T121'] && head_umls.type_ids
            and ug.relationship=''
            and (
                (select array_agg(word_ngrams) from word_ngrams(ug.tail_name, 1)) && head_umls.synonyms
                OR
                (select array_agg(word_ngrams) from word_ngrams(ug.tail_name, 2)) && head_umls.synonyms
                OR
                (select array_agg(word_ngrams) from word_ngrams(ug.tail_name, 3)) && head_umls.synonyms
            )
            and ug.head_id<>ug.tail_id
            AND ug2.head_id is null
        """
        await client.execute_raw(query)

    @staticmethod
    async def copy_all():
        """
        Copy all UMLS data
        """
        await UmlsLoader.create_umls_lookup()
        await UmlsLoader.copy_relationships()
        await UmlsLoader.add_missing_relationships()

    @staticmethod
    async def post_doc_finalize_checksum():
        """
        Quick UMLS checksum
        """
        client = await prisma_client(300)
        checksums = {
            "levels": f"SELECT level, COUNT(*) FROM umls group by level",
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"UMLS Load checksum {key}: {result}")
        return

    @staticmethod
    async def post_doc_finalize():
        """
        To be run after initial UMLS, biomedical entity, and doc loads
        (since it depends upon those being present)
        """
        await UmlsLoader.set_ontology_levels()
        await UmlsLoader.post_doc_finalize_checksum()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            UMLS ETL
            Usage: python3 -m data.etl.entity.biomedical_entity.umls.load_umls [--post-doc-finalize]
            """
        )
        sys.exit()

    if "--post-doc-finalize" in sys.argv:
        asyncio.run(UmlsLoader().post_doc_finalize())
    else:
        asyncio.run(UmlsLoader.copy_all())
