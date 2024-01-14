"""
Abstract document ETL class
"""
from abc import abstractmethod
from prisma import Prisma
import logging

from system import initialize

initialize()

from data.etl.types import BiomedicalEntityLoadSpec
from typings.documents.common import EntityMapType


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DocumentEtl:
    def __init__(self, document_type: str):
        self.document_type = document_type

    @abstractmethod
    async def get_entity_specs(self) -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities
        """
        raise NotImplementedError

    @abstractmethod
    async def copy_documents(self):
        raise NotImplementedError

    async def link_canonical(self):
        """
        - Link mapping tables "intervenable", "indicatable" and "ownable" to canonical entities
        - add instance_rollup and category_rollups

        All biomedical entities must exist first, as well as "umls"
        """
        async with Prisma(http={"timeout": None}) as db:
            bioent_tables = EntityMapType.__dict__.values()
            for bet in bioent_tables:
                await db.execute_raw(
                    f"""
                    UPDATE {bet}
                    SET
                        entity_id=entity_synonym.entity_id,
                        canonical_name=biomedical_entity.name,
                        canonical_type=biomedical_entity.entity_type,
                        category_rollup=umls_category_rollup.preferred_name,
                        instance_rollup=umls_instance_rollup.preferred_name
                    FROM entity_synonym,
                        biomedical_entity,
                        _entity_to_umls as etu
                        umls,
                        umls_instance_rollup,
                        umls_category_rollup
                    WHERE {bet}.name=entity_synonym.term
                    AND entity_synonym.entity_id=biomedical_entity.id
                    AND biomedical_entity.id=etu."A"
                    AND umls.id=etu."B"
                    AND umls_instance_rollup.id=umls.instance_rollup_id
                    AND umls_category_rollup.id=umls.category_rollup_id
                    """
                )

            await db.execute_raw(
                f"""
                UPDATE ownable
                SET
                    owner_id=owner_synonym.owner_id,
                    canonical_name=owner.name,
                    instance_rollup=biomedical_entity.name -- todo,
                    category_rollup=biomedical_entity.name -- todo
                FROM owner_synonym, owner
                WHERE ownable.name=owner_synonym.term
                AND owner_synonym.owner_id=owner.id;
                """
            )

    async def add_counts(self):
        """
        add counts to biomedical_entity & owner (used for autocomplete ordering)
        """
        async with Prisma(http={"timeout": None}) as db:
            # add counts to biomedical_entity & owner
            ent_tables = {
                "intervenable": {
                    "canonical_table": "biomedical_entity",
                    "id_field": "entity_id",
                },
                "indicatable": {
                    "canonical_table": "biomedical_entity",
                    "id_field": "entity_id",
                },
                "ownable": {"canonical_table": "owner", "id_field": "owner_id"},
            }
            for table, info in ent_tables.items():
                await db.execute_raw(
                    f"""
                    CREATE TEMP TABLE temp_count(id int, count int);
                    INSERT INTO temp_count (id, count) SELECT {info["id_field"]} as id, count(*) FROM {table} GROUP BY {info["id_field"]};
                    UPDATE {info["canonical_table"]} ct SET count=temp_count.count FROM temp_count WHERE temp_count.id=ct.id;
                    DROP TABLE IF EXISTS temp_count;
                    """
                )

    async def copy_all(self):
        db = Prisma(auto_register=True, http={"timeout": None})
        await db.connect()
        await self.copy_documents()
        await self.link_canonical()
        await self.add_counts()
        await db.disconnect()
