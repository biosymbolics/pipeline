"""
Abstract document ETL class
"""

from abc import abstractmethod
import asyncio
import logging


from clients.low_level.prisma import prisma_client
from data.etl.base_etl import BaseEtl
from data.etl.types import BiomedicalEntityLoadSpec
from system import initialize
from typings.documents.common import DocType, VectorizableRecordType
from utils.classes import overrides

initialize()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseDocumentEtl(BaseEtl):
    def __init__(self, document_type: DocType, source_db: str):
        super().__init__(
            record_type=VectorizableRecordType(document_type), source_db=source_db
        )

    @abstractmethod
    def entity_specs(self) -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_all(self):
        raise NotImplementedError

    @abstractmethod
    async def copy_documents(self, is_update: bool = False):
        raise NotImplementedError

    @overrides(BaseEtl)
    async def copy_all(self, is_update: bool = False):
        if is_update:
            logger.info("Deleting documents in order to re-create")
            await self.delete_all()
        logger.info("Coping documents...")
        await self.copy_documents()

        logger.info("Coping vectors...")
        await self.copy_vectors()
        await self.checksum()

    @overrides(BaseEtl)
    async def checksum(self):
        """
        Quick checksum
        TODO: raise exceptions if wrong
        """
        client = await prisma_client(300)
        checksums = {
            self.record_type.name: f"SELECT COUNT(*) FROM {self.record_type.name}",
            "owners": f"""
                SELECT COUNT(*) as count, COUNT(distinct owner_id) as distinct_count
                FROM ownable WHERE {self.record_type.name}_id IS NOT NULL
            """,
            "indications": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM indicatable WHERE {self.record_type.name}_id IS NOT NULL
            """,
            "interventions": f"""
                SELECT COUNT(*) as count, COUNT(distinct entity_id) as distinct_count
                FROM intervenable WHERE {self.record_type.name}_id IS NOT NULL
            """,
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"{self.record_type.name} load checksum {key}: {result[0]}")
