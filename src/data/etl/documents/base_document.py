"""
Abstract document ETL class
"""

from abc import abstractmethod
import asyncio
import logging
from typing import Literal


from clients.low_level.prisma import prisma_client
from data.etl.types import BiomedicalEntityLoadSpec
from system import initialize
from typings.documents.common import DocType

initialize()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseDocumentEtl:
    def __init__(self, document_type: DocType, source_db: str):
        self.document_type = document_type
        self.source_db = source_db

    @abstractmethod
    def entity_specs(self) -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_documents(self):
        raise NotImplementedError

    @abstractmethod
    async def copy_documents(self, is_update: bool = False):
        raise NotImplementedError

    async def copy_all(self, is_update: bool = False):
        if is_update:
            logger.info("Deleting documents in order to re-create")
            await self.delete_documents()
        logger.info("Coping documents...")
        await self.copy_documents()

        if self.document_type != DocType.regulatory_approval:
            await self.copy_vectors()

        await self.checksum()

    async def copy_vectors(self):
        """
        Copy vectors to document tables
        """
        client = await prisma_client(1200)
        queries = [
            "CREATE EXTENSION IF NOT EXISTS dblink",
            f"DROP INDEX IF EXISTS {self.document_type}_vector",
            f"""
                UPDATE {self.document_type.name} set vector = v.vector
                FROM dblink(
                    'dbname={self.source_db}',
                    'SELECT id, vector FROM {self.document_type.name}_vectors'
                ) AS v(id TEXT, vector vector(768))
                WHERE {self.document_type.name}.id=v.id;
            """,
            # https://github.com/pgvector/pgvector?tab=readme-ov-file#index-build-time
            "SET maintenance_work_mem = '5GB'",
            f"""
                CREATE INDEX {self.document_type.name}_vector ON {self.document_type.name}
                USING hnsw (vector vector_l2_ops)
            """,
        ]
        for query in queries:
            await client.execute_raw(query)

    async def checksum(self):
        """
        Quick checksum
        TODO: raise exceptions if wrong
        """
        client = await prisma_client(300)
        checksums = {
            self.document_type.name: f"SELECT COUNT(*) FROM {self.document_type.name}",
            "owners": f"""
                SELECT COUNT(*) as count, COUNT(distinct owner_id) as distinct_count
                FROM ownable WHERE {self.document_type.name}_id IS NOT NULL
            """,
            "indications": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM indicatable WHERE {self.document_type.name}_id IS NOT NULL
            """,
            "interventions": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM intervenable WHERE {self.document_type.name}_id IS NOT NULL
            """,
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(
                f"{self.document_type.name} load checksum {key}: {result[0]}"
            )
