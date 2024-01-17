"""
Abstract document ETL class
"""
from abc import abstractmethod
from prisma import Prisma
import logging

from system import initialize

initialize()

from data.etl.types import BiomedicalEntityLoadSpec


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseDocumentEtl:
    def __init__(self, document_type: str):
        self.document_type = document_type

    @abstractmethod
    def entity_specs(self) -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities
        """
        raise NotImplementedError

    @abstractmethod
    async def copy_documents(self):
        raise NotImplementedError

    async def copy_all(self):
        await self.copy_documents()
