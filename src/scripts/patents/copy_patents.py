"""
Patents ETL script
"""
import sys
import asyncio
import logging
from typing import Literal, Sequence
from prisma.enums import BiomedicalEntityType
from prisma.models import Indicatable, Intervenable, Ownable, Patent
from prisma.types import PatentCreateInput, PatentUpdateInput

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    ETL_BASE_DATABASE_URL,
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from constants.umls import LegacyDomainType
from data.etl.biomedical_entity import BiomedicalEntityEtl
from data.etl.document import DocumentEtl
from data.etl.types import RelationConnectInfo, RelationIdFieldMap
from scripts.patents.constants import GPR_ANNOTATIONS_TABLE


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SOURCE_DB = f"{ETL_BASE_DATABASE_URL}/patents"


# if len(grouped_synonyms[0][1]) > MIN_CANONICAL_NAME_COUNT:
canonical_sql = "select id, canonical_name, preferred_name, instance_rollup, category_rollup from umls_lookup"


def get_patent_mapping_entities(domains: Sequence[str]) -> str:
    return f"""
        SELECT publication_number,
            s.term as term,
            s.id as id,
            domain,
            max(source) as source,
            min(character_offset_start) as character_offset_start,
            coalesce(max(t.instance_rollup), s.term) as instance_rollup,
            coalesce(max(t.category_rollup), s.term) as category_rollup
        from (
            SELECT
                publication_number,
                (CASE WHEN map.term is null THEN lower(original_term) ELSE map.term end) as term,
                (CASE WHEN map.id is null THEN lower(original_term) ELSE map.id end) as id,
                domain,
                source,
                character_offset_start,
                FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
                LEFT JOIN synonym_map map ON LOWER(original_term) = map.synonym
            ) s
            LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
            group by publication_number, s.term, s.id, domain
        """


def get_patent_mapping_entities_from_gpr():
    return f"""
        SELECT publication_number,
            s.term as term,
            s.id as id,
            domain,
            min(character_offset_start) as character_offset_start,
            coalesce(max(t.instance_rollup), s.term) as instance_rollup,
            coalesce(max(t.category_rollup), s.term) as category_rollup
        from (
            SELECT
                publication_number,
                (CASE WHEN map.term is null THEN lower(preferred_name) ELSE map.term end) as term,
                (CASE WHEN map.id is null THEN lower(preferred_name) ELSE map.id end) as id,
                domain,
                character_offset_start
                FROM {GPR_ANNOTATIONS_TABLE}
                LEFT JOIN synonym_map map ON LOWER(preferred_name) = map.synonym
            ) s
            LEFT JOIN terms t ON s.id = t.id AND t.id <> ''
            group by publication_number, s.term, s.id, domain
    """


def get_patent_entity_sql(domains: list[LegacyDomainType]) -> str:
    """
    Get entities from biosym annotations table for creation of biomedical entities

    Args:
        domains: domains to copy
    """
    sql_domain_tuple = "(" + ",".join([f"'{d}'" for d in domains]) + ")"
    biosym_sql = f"""
        SELECT lower(original_term) as term
        FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
        WHERE domain in {sql_domain_tuple}
        GROUP BY lower(original_term)
    """

    if "diseases" in domains:
        # add in gpr annotations (that is, annotations from google; we only kept diseases)
        sql = f"""
            SELECT distinct term from (
                SELECT lower(preferred_name) as term
                FROM {GPR_ANNOTATIONS_TABLE}
                WHERE domain='diseases'
                GROUP BY lower(preferred_name)

                UNION ALL

                {biosym_sql}
            ) t
        """
        return sql

    return biosym_sql


class PatentEtl(DocumentEtl):
    @staticmethod
    def get_source_sql():
        return f"""
            SELECT *
            FROM applications
            LEFT JOIN (
                SELECT
                    publication_number,
                    array_agg(original_term) as attributes
                from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
                where domain='attributes'
                group by publication_number
            ) attributes ON applications.publication_number = attributes.publication_number
        """

    async def _copy_entities(
        self,
        domains: list[LegacyDomainType],
        type: BiomedicalEntityType,
        type_type: Literal["override", "default"] = "default",
    ):
        """
        Create entities

        Args:
            domains: domains to copy
            type: entity type
            type_type: type of type - either "override" (forcing) or "default" (only if no type found via canonicalization)
        """
        source_sql = get_patent_entity_sql(domains)
        source_records = await PsqlDatabaseClient(SOURCE_DB).select(query=source_sql)

        source_map = {
            sr["term"]: {
                "synonyms": [sr["term"]],
                "type" if type_type == "override" else "default": type,
            }
            for sr in source_records[0:10]
        }

        terms_to_insert = list(source_map.keys())
        terms_to_canonicalize = terms_to_insert

        await BiomedicalEntityEtl(
            "CompositeCandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
        ).create_records(terms_to_canonicalize, terms_to_insert, source_map=source_map)

    async def copy_indications(self):
        """
        Create indication entity records
        """
        await self._copy_entities(
            ["diseases"], BiomedicalEntityType.DISEASE, type_type="override"
        )

    async def copy_interventions(self):
        """
        Create intervention entity records
        """
        await self._copy_entities(
            [
                "biologics",
                "compounds",
                "devices",
                "procedures",
                "mechanisms",
            ],
            BiomedicalEntityType.OTHER,  # TODO: "INTERVENTION"?
            type_type="default",
        )

    async def copy_documents(self):
        """
        Create regulatory approval records
        """
        records = await PsqlDatabaseClient(SOURCE_DB).select(
            query=PatentEtl.get_source_sql()
        )

        # create patent records
        for p in records:
            data = {
                "id": p["publication_number"],
                "abstract": p["abstract"],
                "application_number": p["application_number"],
                "attributes": p["attributes"],
                # assignees (relation)
                "claims": p["claims"],
                "country_code": p["country"],
                "embeddings": p["embeddings"],
                "ipc_codes": p["ipc_codes"],
                # indications (relation)
                # interventions (relation)
                # inventors (relation)
                "other_ids": p["all_publication_numbers"],
                "priority_date": p["priority_date"],
                "similar_patents": {"set": [{"id": sp} for sp in p["similar_patents"]]},
                "title": p["title"],
                "url": p["url"],
            }
            await Patent.prisma().upsert(
                where={"id": p["publication_number"]},
                data={
                    "create": PatentCreateInput(**data),
                    "update": PatentUpdateInput(**data),
                },
            )

        # create assignee owner records
        await Ownable.prisma().create_many(
            data=[
                {
                    "name": a,
                    "is_primary": True,
                    "assignee_patent_id": p["id"],
                }
                for p in records
                for a in p["assignees"]
            ],
            skip_duplicates=True,
        )

        # create inventor owner records
        await Ownable.prisma().create_many(
            data=[
                {
                    "name": i,
                    "is_primary": True,
                    "inventor_patent_id": p["id"],
                }
                for p in records
                for i in p["inventors"]
            ],
            skip_duplicates=True,
        )

        # create "indicatable" records, those that map approval to a canonical indication
        await Indicatable.prisma().create_many(
            data=[{"name": "", "patent_id": p["id"]} for p in records]
        )

        # create "intervenable" records, those that map approval to a canonical intervention
        await Intervenable.prisma().create_many(
            data=[
                {
                    "name": "",
                    "is_primary": True,
                    "regulatory_approval_id": p["id"],
                }
                for p in records
            ]
        )


async def main():
    await PatentEtl(document_type="patent").copy_all()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.patents.copy_patents
            Copies patents data to biosym
            """
        )
        sys.exit()

    asyncio.run(main())
