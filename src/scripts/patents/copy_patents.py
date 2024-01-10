"""
Patents ETL script
"""
import sys
import asyncio
import logging
from typing import Literal, Sequence
from prisma import Prisma
from prisma.enums import BiomedicalEntityType, Source
from prisma.models import Indicatable, Intervenable, Ownable, Patent
from prisma.types import PatentCreateInput

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
INTERVENTION_DOMAINS: list[LegacyDomainType] = [
    "biologics",
    "compounds",
    "devices",
    "procedures",
    "mechanisms",
]

# if len(grouped_synonyms[0][1]) > MIN_CANONICAL_NAME_COUNT:
canonical_sql = "select id, canonical_name, preferred_name, instance_rollup, category_rollup from umls_lookup"


def get_patent_mapping_entities_sql(domains: Sequence[str]) -> str:
    """
    Get sql for mapping entities (i.e. intervenable, indicatable)
    """
    biosym_sql = f"""
            SELECT
                publication_number as id,
                lower(original_term) as term,
                min(character_offset_start) as mention_index
            FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
            WHERE domain in ('{"','".join(domains)}')
            GROUP BY publication_number, lower(original_term)
        """
    if "diseases" in domains:
        sql = f"""
            SELECT term, id, min(mention_index) as mention_index FROM (
                SELECT
                    publication_number as id,
                    lower(preferred_name) as term,
                    min(character_offset_start) as mention_index
                FROM {GPR_ANNOTATIONS_TABLE}
                WHERE domain='diseases'
                GROUP BY publication_number, lower(preferred_name)

                UNION ALL

                {biosym_sql}
            ) t GROUP BY term, id
        """
        return sql

    return biosym_sql


def get_patent_entity_sql(domains: list[LegacyDomainType]) -> str:
    """
    Get entities from biosym annotations table for creation of biomedical entities

    Args:
        domains: domains to copy
    """
    biosym_sql = f"""
        SELECT lower(original_term) as term
        FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
        WHERE domain in ('{"','".join(domains)}')
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
    def get_source_sql(fields: list[str]):
        return f"""
            SELECT {','.join(fields)}
            FROM applications
            LEFT JOIN (
                SELECT
                    publication_number,
                    array_agg(original_term) as attributes
                from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
                where domain='attributes'
                group by publication_number
            ) attributes ON applications.publication_number = attributes.publication_number
            LEFT JOIN (
                SELECT
                    publication_number,
                    array_agg(original_term) as annotations
                from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
                where domain<>'attributes'
                group by publication_number
            ) annotations ON applications.publication_number = annotations.publication_number
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

        async def handle_batch(batch: list[dict]):
            # dedup and map to source
            source_map = {
                sr["term"]: {
                    "synonyms": [sr["term"]],
                    "type" if type_type == "override" else "default_type": type,
                }
                for sr in batch
            }

            terms = list(source_map.keys())

            await BiomedicalEntityEtl(
                "CompositeCandidateSelector",
                relation_id_field_map=RelationIdFieldMap(
                    synonyms=RelationConnectInfo(
                        source_field="synonyms", dest_field="term", input_type="create"
                    ),
                ),
                non_canonical_source=Source.BIOSYM,
            ).create_records(terms, source_map=source_map)

        source_sql = get_patent_entity_sql(domains)
        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=source_sql, handle_result_batch=handle_batch, batch_size=200000
        )

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
            INTERVENTION_DOMAINS,
            BiomedicalEntityType.OTHER,  # TODO: "INTERVENTION"?
            type_type="default",
        )

    async def copy_documents(self):
        """
        Create regulatory approval records
        """

        async def handle_batch(batch: list[dict]) -> bool:
            # create patent records
            await Patent.prisma().create_many(
                data=[
                    PatentCreateInput(
                        **{
                            "id": p["id"],
                            "abstract": p["abstract"],
                            "application_number": p["application_number"],
                            "attributes": p["attributes"] or [],
                            # assignees (relation)
                            "claims": p["claims"],
                            "country_code": p["country"],
                            # "embeddings": p["embeddings"], # updated below (https://github.com/prisma/prisma/issues/18442)
                            "ipc_codes": p["ipc_codes"],
                            # indications (relation)
                            # interventions (relation)
                            # inventors (relation)
                            "other_ids": p["all_publication_numbers"],
                            "priority_date": p["priority_date"],
                            "similar_patents": p["similar_patents"] or [],
                            "text_for_search": f"{p['title']} {p['abstract']} {' '.join(p['assignees'])} {' '.join(p['annotations'])}",  # TODO: add canonicalized annotations!!!
                            "title": p["title"],
                            "url": p["url"],
                        }
                    )
                    for p in batch
                ],
                skip_duplicates=True,
            )

            # TODO: do in bulk for perf
            async with Prisma(http={"timeout": None}) as db:
                # sigh https://github.com/prisma/prisma/issues/18442
                for p in batch:
                    await db.execute_raw(
                        f"UPDATE patent SET embeddings = '{p['embeddings']}' where id = '{p['id']}';"
                    )

            # create assignee owner records
            await Ownable.prisma().create_many(
                data=[
                    {
                        "name": a.lower(),
                        "canonical_name": a.lower(),  # may be overwritten later
                        "is_primary": True,
                        "patent_id": p["id"],
                    }
                    for p in batch
                    for a in p["assignees"]
                ],
                skip_duplicates=True,
            )

            # create inventor owner records
            await Ownable.prisma().create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # may be overwritten later
                        "is_primary": True,
                        "inventor_patent_id": p["id"],
                    }
                    for p in batch
                    for i in p["inventors"]
                ],
                skip_duplicates=True,
            )

            return True

        source_fields = [
            "applications.publication_number as id",
            "abstract",
            "all_publication_numbers",
            "application_number",
            "assignees",
            "attributes",
            "claims",
            "country",
            "embeddings",
            "ipc_codes",
            "inventors",
            "priority_date::TIMESTAMP as priority_date",
            "similar_patents",
            "title",
            "url",
        ]
        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=self.get_source_sql(source_fields),
            batch_size=10000,
            handle_result_batch=handle_batch,  # type: ignore
        )

        # create "indicatable" records, those that map approval to a canonical indication
        indicatable_records = await PsqlDatabaseClient(SOURCE_DB).select(
            query=get_patent_mapping_entities_sql(["diseases"])
        )
        await Indicatable.prisma().create_many(
            data=[
                {
                    "is_primary": False,  # TODO
                    "mention_index": ir["mention_index"],
                    "name": ir["term"],
                    "canonical_name": ir["term"],  # overwritten later
                    "patent_id": ir["id"],
                }
                for ir in indicatable_records
            ],
            skip_duplicates=True,
        )

        # create "intervenable" records, those that map approval to a canonical intervention
        intervenable_records = await PsqlDatabaseClient(SOURCE_DB).select(
            query=get_patent_mapping_entities_sql(INTERVENTION_DOMAINS)
        )
        await Intervenable.prisma().create_many(
            data=[
                {
                    "is_primary": False,  # TODO
                    "mention_index": ir["mention_index"],
                    "name": ir["term"],
                    "canonical_name": ir["term"],  # overwritten later
                    "patent_id": ir["id"],
                }
                for ir in intervenable_records
            ],
            skip_duplicates=True,
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
