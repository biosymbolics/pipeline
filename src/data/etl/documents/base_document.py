"""
Abstract document ETL class
"""

from abc import abstractmethod
import asyncio
import logging
from typing import Literal
from clients.low_level.prisma import prisma_client

from system import initialize

initialize()

from data.etl.types import BiomedicalEntityLoadSpec


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DocumentType = Literal["patent", "regulatory_approval", "trial"]


class BaseDocumentEtl:
    def __init__(self, document_type: DocumentType):
        self.document_type = document_type

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
        await self.checksum()

    async def checksum(self):
        """
        Quick checksum
        TODO: raise exceptions if wrong
        """
        client = await prisma_client(300)
        checksums = {
            self.document_type: f"SELECT COUNT(*) FROM {self.document_type}",
            "owners": f"SELECT COUNT(*) FROM ownable WHERE {self.document_type}_id IS NOT NULL",
            "indications": f"SELECT COUNT(*) FROM indicatable WHERE {self.document_type}_id IS NOT NULL",
            "interventions": f"SELECT COUNT(*) FROM intervenable WHERE {self.document_type}_id IS NOT NULL",
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"{self.document_type} load checksum {key}: {result[0]}")
