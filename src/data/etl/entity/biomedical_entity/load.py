import asyncio
import sys
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_context
from data.etl.documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


from .biomedical_entity import BiomedicalEntityEtl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    @staticmethod
    async def post_doc_finalize():
        await BiomedicalEntityEtl.post_doc_finalize()


def main():
    prisma_context(300)
    asyncio.run(BiomedicalEntityLoader().copy_all())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.biomedical_entity.load
            """
        )
        sys.exit()

    main()
