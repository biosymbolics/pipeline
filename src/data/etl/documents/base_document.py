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

initialize()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DocumentType = Literal["patent", "regulatory_approval", "trial"]


class BaseDocumentEtl:
    def __init__(self, document_type: DocumentType, source_db: str):
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
        await self.copy_vectors()
        await self.checksum()

    async def copy_vectors(self):
        """
        Copy vectors to document tables
        """
        client = await prisma_client(1200)
        row_count = await client.query_raw(
            f"SELECT COUNT(*) as count FROM {self.document_type}"
        )
        list_count = int(row_count[0]["count"] / 1000)
        queries = [
            "CREATE EXTENSION IF NOT EXISTS dblink",
            f"DROP INDEX IF EXISTS {self.document_type}_vector",
            f"""
                UPDATE {self.document_type} set vector = v.vector
                FROM dblink(
                    'dbname={self.source_db}',
                    'SELECT id, vector FROM {self.document_type}_vectors'
                ) AS v(id TEXT, vector vector(768))
                WHERE {self.document_type}.id=v.id
            """,
            # TODO: switch to hnsw maybe
            # "CREATE INDEX ON patent USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64)",
            # sizing: https://github.com/pgvector/pgvector#ivfflat (rows / 1000)
            f"""
                CREATE INDEX {self.document_type}_vector ON {self.document_type}
                USING ivfflat (vector vector_cosine_ops) WITH (lists = {list_count})
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
            self.document_type: f"SELECT COUNT(*) FROM {self.document_type}",
            "owners": f"""
                SELECT COUNT(*) as count, COUNT(distinct owner_id) as distinct_count
                FROM ownable WHERE {self.document_type}_id IS NOT NULL
            """,
            "indications": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM indicatable WHERE {self.document_type}_id IS NOT NULL
            """,
            "interventions": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM intervenable WHERE {self.document_type}_id IS NOT NULL
            """,
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"{self.document_type} load checksum {key}: {result[0]}")
