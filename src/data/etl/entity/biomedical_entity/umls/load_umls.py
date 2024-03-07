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


from clients.low_level.prisma import batch_update, prisma_client
from clients.low_level.postgres import PsqlDatabaseClient
from constants.umls import BIOMEDICAL_GRAPH_UMLS_TYPES
from data.domain.biomedical.umls import clean_umls_name
from data.etl.base_etl import BaseEtl
from system import initialize
from typings.documents.common import VectorizableRecordType
from utils.classes import overrides

from .vectorize_umls import UmlsVectorizer
from .umls_transform import UmlsAncestorTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BATCH_SIZE = 10000

initialize()


class UmlsLoader(BaseEtl):
    @staticmethod
    def get_source_sql(filters: list[str] | None = None, limit: int | None = None):
        return f"""
            SELECT
                TRIM(entities.cui) AS id,
                TRIM(max(entities.str)) AS name,
                COALESCE(array_agg(distinct semantic_types.tui::text), ARRAY[]::text[]) AS type_ids,
                COALESCE(array_agg(distinct semantic_types.sty::text), ARRAY[]::text[]) AS type_names,
                COALESCE(max(synonyms.terms), ARRAY[]::text[]) as synonyms
            FROM mrconso AS entities
            LEFT JOIN mrsty AS semantic_types ON semantic_types.cui = entities.cui
            LEFT JOIN (
                SELECT ARRAY_AGG(DISTINCT(LOWER(str))) AS terms, cui AS id
                FROM mrconso
                GROUP BY cui
            ) AS synonyms ON synonyms.id = entities.cui
            WHERE entities.lat='ENG' -- english
            AND entities.ts='P' -- preferred terms (according to UMLS)
            AND entities.ispref='Y' -- preferred term (according to UMLS)
            AND entities.stt='PF' -- preferred term (according to UMLS)
            {('AND ' + ' AND '.join(filters)) if filters else ''}
            GROUP BY entities.cui -- because multiple preferred terms (according to UMLS)
            ORDER BY entities.cui ASC
            {'LIMIT ' + str(limit) if limit else ''}
        """

    async def create_umls_lookup(self):
        """
        Create UMLS lookup table

        Creates a table of UMLS entities: id, name, ancestor ids
        """

        client = await prisma_client(600)
        source_sql = UmlsLoader.get_source_sql()

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

        await PsqlDatabaseClient(self.source_db).execute_query(
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

    async def copy_relationships(self):
        """
        Copy relationships from umls to patents

        - Creates a table of UMLS relationships: head_id, head_name, tail_id, tail_name, relationship
        - Limits based on language and semantic type
        """
        source_sql = f"""
            SELECT
                cui1 AS head_id, MAX(head.str) AS head_name,
                cui2 AS tail_id, MAX(tail.str) AS tail_name,
                COALESCE(rela, '') AS relationship
            FROM mrrel
            JOIN mrconso AS head on head.cui = cui1
            JOIN mrconso AS tail on tail.cui = cui2
            JOIN mrsty AS head_semantic_type ON head_semantic_type.cui = cui1
            JOIN mrsty AS tail_semantic_type ON tail_semantic_type.cui = cui2
            WHERE head.lat='ENG'
            AND head.ts='P'
            AND head.ispref='Y'
            AND tail.lat='ENG'
            AND tail.ts='P'
            AND tail.ispref='Y'
            AND head_semantic_type.tui IN {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
            AND tail_semantic_type.tui IN {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
            GROUP BY head_id, tail_id, relationship
        """

        async def handle_batch(batch):
            logger.info("Creating %s UmlsGraph records", len(batch))
            await UmlsGraph.prisma().create_many(
                data=[UmlsGraphRecord(**r) for r in batch],
                skip_duplicates=True,
            )

        await PsqlDatabaseClient(self.source_db).execute_query(
            query=source_sql, batch_size=BATCH_SIZE, handle_result_batch=handle_batch
        )

    @staticmethod
    async def add_missing_relationships():
        """
        Add missing UMLS relationships

        TODO:
        - handle similar synonyms, e.g. normalize for plurals and dashes (ex C3864925 vs C3864917)
        """
        client = await prisma_client(600)
        # if no 'isa' relationship exists for the tail
        # AND the tail name is a subset of the head synonyms (e.g. XYZ inhibitor -> XYZ)
        # then mark the relationship as 'isa'
        create_fun_query = """
            CREATE or REPLACE FUNCTION word_ngrams(str text, n int)
            RETURNS setof text language plpgsql as $$
            DECLARE
                i int;
                arr text[];
            BEGIN
                arr := regexp_split_to_array(str, '[^[:alnum:]]+');
                for i in 1 .. cardinality(arr)-n+1 loop
                    return next array_to_string(arr[i : i+n-1], ' ');
                end loop;
            END $$;
        """
        # TODO: remove s, dash, and presense of non-distinguishing words like "receptor", "gene" and "antibody"
        await client.execute_raw(create_fun_query)
        query = """
            UPDATE umls_graph SET relationship='isa'
            FROM umls head_umls
            WHERE head_id=head_umls.id
            AND relationship=''
            AND (
                (SELECT array_agg(word_ngrams) FROM word_ngrams(tail_name, 1)) && head_umls.synonyms
                OR
                (SELECT array_agg(word_ngrams) FROM word_ngrams(tail_name, 2)) && head_umls.synonyms
                OR
                (SELECT array_agg(word_ngrams) FROM word_ngrams(tail_name, 3)) && head_umls.synonyms
            )
            AND head_id<>tail_id
            AND NOT EXISTS (
                SELECT 1 FROM umls_graph
                WHERE umls_graph.relationship='isa'
                AND umls_graph.tail_id=tail_id
                LIMIT 1
            )
        """
        await client.execute_raw(query)

    @staticmethod
    async def remove_duplicates():
        """
        Make certain duplicate UMLS records unavailable for rollup
        (based on synonyms)

        TODO
        - cytokine receptor gene -> cytokine
        - add more types!!
        """
        query = """
            UPDATE umls SET level='NA'
            WHERE id IN (
                SELECT id FROM umls, unnest(synonyms) synonym
                WHERE NOT 'T116'=ANY(type_ids) -- Amino Acid, Peptide, or Protein
                AND synonym IN (
                    SELECT synonym
                    FROM umls, UNNEST(synonyms) synonym
                    WHERE 'T116'=ANY(type_ids)
                    GROUP BY synonym HAVING count(*) > 1
                )
            )
        """
        client = await prisma_client(600)
        await client.execute_raw(query)

    @staticmethod
    @overrides(BaseEtl)
    async def checksum():
        """
        Quick UMLS checksum
        """
        client = await prisma_client(300)
        checksums = {
            "levels": f"SELECT level, COUNT(*) FROM umls GROUP BY level",
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"UMLS Load checksum {key}: {result}")
        return

    @overrides(BaseEtl)
    async def copy_all(self):
        """
        Copy all UMLS data
        """
        await UmlsVectorizer()()

        await self.create_umls_lookup()
        await self.copy_vectors()
        await self.copy_relationships()
        await self.add_missing_relationships()

    @staticmethod
    async def post_doc_finalize():
        """
        To be run after initial UMLS, biomedical entity, and doc loads
        (since it depends upon those being present)
        """
        await UmlsLoader.set_ontology_levels()
        await UmlsLoader.remove_duplicates()
        await UmlsLoader.checksum()


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
        asyncio.run(UmlsLoader.post_doc_finalize())
    else:
        asyncio.run(
            UmlsLoader(
                record_type=VectorizableRecordType.umls,
                source_db="umls",
            ).copy_all()
        )
