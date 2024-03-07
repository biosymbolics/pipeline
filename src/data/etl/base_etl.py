"""
Abstract document ETL class
"""

from abc import abstractmethod
import asyncio
import logging


from clients.low_level.prisma import prisma_client
from data.etl.types import BiomedicalEntityLoadSpec
from system import initialize
from typings.documents.common import DocType, VectorizableRecordType

initialize()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseEtl:
    def __init__(self, record_type: VectorizableRecordType, source_db: str):
        self.record_type = record_type
        self.source_db = source_db

    async def copy_all(self, is_update: bool = False):
        raise NotImplementedError

    async def copy_vectors(self):
        """
        Copy vectors to document tables
        """
        client = await prisma_client(2400)
        queries = [
            "CREATE EXTENSION IF NOT EXISTS dblink",
            f"DROP INDEX IF EXISTS {self.record_type}_vector",
            f"""
                UPDATE {self.record_type.name} set vector = v.vector
                FROM dblink(
                    'dbname={self.source_db}',
                    'SELECT id, vector FROM {self.record_type.name}_vectors'
                ) AS v(id TEXT, vector vector(768))
                WHERE {self.record_type.name}.id=v.id;
            """,
            # https://github.com/pgvector/pgvector?tab=readme-ov-file#index-build-time
            "SET maintenance_work_mem = '5GB'",
            f"""
                CREATE INDEX {self.record_type.name}_vector ON {self.record_type.name}
                USING hnsw (vector vector_cosine_ops)
            """,
        ]
        for query in queries:
            await client.execute_raw(query)

    @staticmethod
    async def finalize():
        raise NotImplementedError

    @staticmethod
    async def checksum():
        raise NotImplementedError
