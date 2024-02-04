import asyncio
import sys
import logging
import httpx
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_client
from data.etl.documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


from .biomedical_entity import BiomedicalEntityEtl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def entity_checksum():
    """
    Quick entity checksum
    """
    client = await prisma_client(300)
    checksums = {
        "comprised_of": f"SELECT COUNT(*) FROM _entity_comprised_of",
        "parents": f"SELECT COUNT(*) FROM _entity_to_parent",
        "biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity",
        "priority_biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity where is_priority=true",
        "umls_biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity, umls where umls.id=biomedical_entity.canonical_id",
    }
    results = await asyncio.gather(
        *[client.query_raw(query) for query in checksums.values()]
    )
    for key, result in zip(checksums.keys(), results):
        logger.warning(f"Load checksum {key}: {result[0]}")
    return


class BiomedicalEntityLoader:
    @staticmethod
    async def copy_all():
        """
        Copies all biomedical entities based on specs pulled from each document type

        NOTE: is slow due to UMLS linking (5-8 hours?)
        """
        # patent_specs = PatentLoader.entity_specs()
        regulatory_approval_specs = RegulatoryApprovalLoader.entity_specs()
        # trial_specs = TrialLoader.entity_specs()

        specs = regulatory_approval_specs  # + patent_specs + trial_specs

        for spec in specs:
            records = await PsqlDatabaseClient(spec.database).select(spec.sql)
            source_map = spec.get_source_map(records)
            terms = spec.get_terms(source_map)
            to_canonicalize = spec.get_terms_to_canonicalize(source_map)
            logger.info("ETLing %s terms", len(to_canonicalize[0]))
            await BiomedicalEntityEtl(
                candidate_selector=spec.candidate_selector,
                relation_id_field_map=spec.relation_id_field_map,
                non_canonical_source=spec.non_canonical_source,
            ).copy_all(terms, *to_canonicalize, source_map)

        await BiomedicalEntityEtl.pre_doc_finalize()
        await entity_checksum()
        logger.info("Biomedical entity load complete")

    @staticmethod
    async def post_doc_finalize():
        await BiomedicalEntityEtl.post_doc_finalize()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.biomedical_entity.biomedical_entity_load [--post-doc-finalize]
            """
        )
        sys.exit()

    if "--post-doc-finalize" in sys.argv:
        asyncio.run(BiomedicalEntityLoader().post_doc_finalize())
    else:
        asyncio.run(BiomedicalEntityLoader().copy_all())
