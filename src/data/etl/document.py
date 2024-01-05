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

    async def link_mapping_tables(self):
        """
        Link mapping tables "intervenable", "indicatable" and "ownable" to canonical entities
        """
        async with Prisma() as db:
            bioent_tables = ["intervenable", "indicatable"]
            for bet in bioent_tables:
                await db.execute_raw(
                    f"""
                    UPDATE {bet}
                    SET entity_id=entity_synonym.entity_id
                    FROM entity_synonym
                    WHERE {bet}.name=entity_synonym.term;
                    """
                )

            await db.execute_raw(
                f"""
                UPDATE ownable
                SET owner_id=synonym.owner_id
                FROM owner_synonym
                WHERE owner.name=owner_synonym.term;
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
        await self.link_mapping_tables()
        await self.copy_indications()
        await self.create_search_index()
        await db.disconnect()
