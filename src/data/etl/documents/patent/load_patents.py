"""
Patents ETL script
"""

import sys
import asyncio
import logging
from typing import Sequence
from prisma.enums import BiomedicalEntityType
from prisma.models import Indicatable, Intervenable, Ownable, Patent
from prisma.types import PatentCreateInput
from pydash import uniq

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_client
from constants.core import (
    GPR_ANNOTATIONS_TABLE,
    SOURCE_BIOSYM_ANNOTATIONS_TABLE,
    WORKING_BIOSYM_ANNOTATIONS_TABLE,
)
from constants.documents import PATENT_WEIGHT_MULTIPLIER
from constants.umls import LegacyDomainType
from data.etl.types import BiomedicalEntityLoadSpec
from typings import DocType
from utils.classes import overrides

from ..base_document_etl import BaseDocumentEtl


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


INTERVENTION_DOMAINS: list[LegacyDomainType] = [
    "biologics",
    "compounds",
    "devices",
    "diagnostics",
    "procedures",
    "mechanisms",
]


PATENT_SOURCE_FIELDS = [
    "applications.publication_number as id",
    "abstract",
    "all_publication_numbers",
    "application_number",
    "assignees",
    "attributes",
    "ARRAY_TO_STRING(COALESCE(claims, ARRAY[]::text[]), '\n') as claims",
    "country",
    "family_id",
    "ipc_codes",
    "inventors",
    "priority_date::TIMESTAMP as priority_date",
    "similar_patents",
    "title",
    "url",
]


def get_mapping_entities_sql(domains: Sequence[str]) -> str:
    """
    Get sql for mapping entities (i.e. intervenable, indicatable)
    """
    biosym_sql = f"""
            SELECT
                publication_number as id,
                lower(term) as term,
                min(character_offset_start) as mention_index
            FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
            WHERE domain in ('{"','".join(domains)}')
            -- temp hack
            AND publication_number in
                (select publication_number from applications where publication_number={WORKING_BIOSYM_ANNOTATIONS_TABLE}.publication_number)
            GROUP BY publication_number, lower(term)
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
                AND publication_number in
                    (select publication_number from applications where publication_number={GPR_ANNOTATIONS_TABLE}.publication_number)
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
                    array_agg(term) as attributes
                from {SOURCE_BIOSYM_ANNOTATIONS_TABLE}
                where domain='attributes'
                group by publication_number
            ) attributes ON applications.publication_number = attributes.publication_number
            LEFT JOIN (
                SELECT
                    publication_number,
                    array_agg(term) as annotations
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
            SELECT distinct lower(term) as term --, vector
            FROM {WORKING_BIOSYM_ANNOTATIONS_TABLE}
            WHERE domain in ('{"','".join(domains)}')
        """

        if "diseases" in domains:
            # add in gpr annotations (that is, annotations from google; we only kept diseases)
            sql = f"""
                SELECT distinct term from (
                    SELECT distinct lower(preferred_name) as term --, ARRAY[]::float[] as vector
                    FROM {GPR_ANNOTATIONS_TABLE}
                    WHERE domain='diseases'

                    UNION ALL

                    {biosym_sql}
                ) t
            """
            return sql

        return biosym_sql

    @overrides(BaseDocumentEtl)
    @staticmethod
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="patents",
            get_source_map=lambda recs: {
                rec["term"]: {
                    "synonyms": [rec["term"]],
                    "type": BiomedicalEntityType.DISEASE,
                    # "vector": rec["vector"],
                }
                for rec in recs
            },
            get_terms_to_canonicalize=lambda sm: (
                list(sm.keys()),
                None,  # [sm[t]["vector"] for t in sm],
            ),
            sql=PatentLoader.get_entity_sql(["diseases"]),
        )
        intervention_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="patents",
            get_source_map=lambda recs: {
                rec["term"]: {
                    "synonyms": [rec["term"]],
                    "default_type": BiomedicalEntityType.OTHER,
                    # "vector": rec["vector"],
                }
                for rec in recs
            },
            get_terms_to_canonicalize=lambda sm: (
                list(sm.keys()),
                None,  # [sm[t]["vector"] for t in sm],
            ),
            sql=PatentLoader.get_entity_sql(INTERVENTION_DOMAINS),
        )
        return [intervention_spec, indication_spec]

    @overrides(BaseDocumentEtl)
    async def delete_all(self):
        """
        Delete all patent records (which should cascade)
        """

        client = await prisma_client(600)
        await Ownable.prisma(client).query_raw(
            "DELETE FROM ownable WHERE patent_id IS NOT NULL or inventor_patent_id IS NOT NULL"
        )
        await Intervenable.prisma(client).query_raw(
            "DELETE FROM intervenable WHERE patent_id IS NOT NULL"
        )
        await Indicatable.prisma(client).query_raw(
            "DELETE FROM indicatable WHERE patent_id IS NOT NULL"
        )
        await Patent.prisma(client).delete_many()

    @overrides(BaseDocumentEtl)
    async def copy_documents(self):
        """
        Create regulatory approval records
        """

        client = await prisma_client(600)

        async def handle_batch(batch: list[dict]) -> bool:
            logger.info("Creating %s patent records", len(batch))
            await Patent.prisma(client).create_many(
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
                            "family_id": p["family_id"],
                            "investment": len(p["all_publication_numbers"])
                            * PATENT_WEIGHT_MULTIPLIER,
                            "ipc_codes": p["ipc_codes"],
                            # indications (relation)
                            # interventions (relation)
                            # inventors (relation)
                            "other_ids": p["all_publication_numbers"],
                            "priority_date": p["priority_date"],
                            "similar_patents": p["similar_patents"] or [],
                            "title": p["title"],
                            "traction": len(p["all_publication_numbers"])
                            * PATENT_WEIGHT_MULTIPLIER,
                            "url": p["url"],
                        }
                    )
                    for p in batch
                ],
                skip_duplicates=True,
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
                    for a in uniq(p["assignees"])
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
                        "category_rollup": i.lower(),
                        "is_primary": True,
                        "inventor_patent_id": p["id"],
                    }
                    for p in batch
                    for i in uniq(p["inventors"])
                ],
                skip_duplicates=True,
            )

            return True

        await PsqlDatabaseClient(self.source_db).execute_query(
            query=self.get_source_sql(PATENT_SOURCE_FIELDS),
            batch_size=10000,
            handle_result_batch=handle_batch,  # type: ignore
        )

        # create "indicatable" records, those that map approval to a canonical indication
        indicatable_records = await PsqlDatabaseClient(self.source_db).select(
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
        intervenable_records = await PsqlDatabaseClient(self.source_db).select(
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


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.patent.load_patents [--update]
            Copies patents data to biosym

            Has many dependencies (see stage1_patents)
            """
        )
        sys.exit()

    is_update = "--update" in sys.argv

    asyncio.run(
        PatentLoader(document_type=DocType.patent, source_db="patents").copy_all(
            is_update
        )
    )
