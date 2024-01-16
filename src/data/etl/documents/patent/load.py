"""
Patents ETL script
"""
import sys
import asyncio
import logging
from typing import Sequence
from prisma import Prisma
from prisma.enums import BiomedicalEntityType
from prisma.models import Indicatable, Intervenable, Ownable, Patent
from prisma.types import PatentCreateInput

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_client
from constants.core import (
    ETL_BASE_DATABASE_URL,
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from constants.umls import LegacyDomainType
from data.etl.types import BiomedicalEntityLoadSpec

from ..base_document import BaseDocumentEtl

GPR_ANNOTATIONS_TABLE = "gpr_annotations"


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


def get_mapping_entities_sql(domains: Sequence[str]) -> str:
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


class PatentLoader(BaseDocumentEtl):
    """
    Load patents and associated entities
    """

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

    @staticmethod
    def get_entity_sql(domains: list[LegacyDomainType]) -> str:
        """
        Get entities from biosym annotations table for creation of biomedical entities

        Args:
            domains: domains to copy
        """
        biosym_sql = f"""
            SELECT distinct lower(original_term) as term
            FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
            WHERE domain in ('{"','".join(domains)}')
        """

        if "diseases" in domains:
            # add in gpr annotations (that is, annotations from google; we only kept diseases)
            sql = f"""
                SELECT distinct term from (
                    SELECT distinct lower(preferred_name) as term
                    FROM {GPR_ANNOTATIONS_TABLE}
                    WHERE domain='diseases'

                    UNION ALL

                    {biosym_sql}
                ) t
            """
            return sql

        return biosym_sql

    @staticmethod
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="patents",
            get_source_map=lambda recs: {
                rec["term"]: {
                    "synonyms": [rec["term"]],
                    "type": BiomedicalEntityType.DISEASE,
                }
                for rec in recs
            },
            sql=PatentLoader.get_entity_sql(["diseases"]),
        )
        intervention_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="patents",
            get_source_map=lambda recs: {
                rec["term"]: {
                    "synonyms": [rec["term"]],
                    "default_type": BiomedicalEntityType.OTHER,
                }
                for rec in recs
            },
            sql=PatentLoader.get_entity_sql(INTERVENTION_DOMAINS),
        )
        return [indication_spec, intervention_spec]

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
                            "title": p["title"],
                            "url": p["url"],
                        }
                    )
                    for p in batch
                ],
                skip_duplicates=True,
            )

            client = await prisma_client(None)
            # sigh https://github.com/prisma/prisma/issues/18442
            for p in batch:
                await client.execute_raw(
                    f"UPDATE patent SET embeddings = '{p['embeddings']}' where id = '{p['id']}';"
                )

            # create assignee owner records
            await Ownable.prisma().create_many(
                data=[
                    {
                        "name": a.lower(),
                        "canonical_name": a.lower(),  # may be overwritten later
                        "instance_rollup": a.lower(),  # may be overwritten later
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
                        "instance_rollup": i.lower(),  # may be overwritten later
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
            query=get_mapping_entities_sql(["diseases"])
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
            query=get_mapping_entities_sql(INTERVENTION_DOMAINS)
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
    await PatentLoader(document_type="patent").copy_all()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.patent.load
            Copies patents data to biosym
            """
        )
        sys.exit()

    asyncio.run(main())
