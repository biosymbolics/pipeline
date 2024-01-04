"""
Abstract document ETL class
"""
from abc import abstractmethod
from prisma import Prisma
import logging

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentEtl:
    def __init__(self, document_type: str):
        self.document_type = document_type

    @abstractmethod
    async def copy_indications(self):
        raise NotImplementedError

    @abstractmethod
    async def copy_interventions(self):
        raise NotImplementedError

    @abstractmethod
    async def copy_documents(self):
        raise NotImplementedError

    async def link_maps(self):
        """
        Link mapping tables like "intervenable" and "indicatable" to canonical entities
        """
        async with Prisma() as db:
            mapping_tables = ["intervenable", "indicatable"]
            for mt in mapping_tables:
                await db.execute_raw(
                    f"""
                    UPDATE {mt}
                    SET entity_id=synonym.entity_id
                    FROM synonym
                    WHERE {mt}.name=synonym.term;
                    """
                )

    async def create_search_index(self):
        """
        create search index (unsupported by Prisma)
        """
        async with Prisma(http={"timeout": None}) as db:
            await db.execute_raw(
                f"""
                UPDATE {self.document_type} SET search = to_tsvector('english', text_for_search)
                """
            )
        await PsqlDatabaseClient().create_indices(
            [
                {
                    "table": self.document_type,
                    "column": "search",
                    "is_gin": True,
                },
            ]
        )

    async def copy_all(self):
        db = Prisma(auto_register=True, http={"timeout": None})
        await db.connect()
        await self.copy_interventions()
        await self.copy_indications()
        await self.copy_documents()
        await self.link_maps()
        await self.copy_indications()
        await self.create_search_index()
        await db.disconnect()
