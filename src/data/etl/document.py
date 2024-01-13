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
        async with Prisma(http={"timeout": None}) as db:
            bioent_tables = ["intervenable", "indicatable"]
            for bet in bioent_tables:
                await db.execute_raw(
                    f"""
                    UPDATE {bet}
                    SET
                        entity_id=entity_synonym.entity_id,
                        canonical_name=biomedical_entity.name,
                        instance_rollup=biomedical_entity.name, -- todo
                        canonical_type=biomedical_entity.entity_type
                    FROM entity_synonym, biomedical_entity
                    WHERE {bet}.name=entity_synonym.term
                    AND entity_synonym.entity_id=biomedical_entity.id;
                    """
                )

            # add counts to biomedical_entity
            await db.execute_raw(
                """
                CREATE TEMP TABLE biomedical_entity_count(
                    entity_id int,
                    count int
                );
                INSERT INTO biomedical_entity_count (entity_id, count)
                    SELECT entity_id, count(*) FROM {bet} GROUP BY entity_id;
                DROP TABLE IF EXISTS biomedical_entity_count;
                """
            )

            await db.execute_raw(
                f"""
                UPDATE ownable
                SET
                    owner_id=owner_synonym.owner_id,
                    canonical_name=owner.name,
                    instance_rollup=biomedical_entity.name -- todo
                FROM owner_synonym, owner
                WHERE ownable.name=owner_synonym.term
                AND owner_synonym.owner_id=owner.id;
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
        await self.copy_documents()
        await self.copy_interventions()
        await self.copy_indications()
        await self.link_mapping_tables()
        await self.create_search_index()
        await db.disconnect()
