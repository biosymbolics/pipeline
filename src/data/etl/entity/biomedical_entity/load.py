import asyncio
import sys

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_context
from data.etl.documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


from .biomedical_entity import BiomedicalEntityEtl


class BiomedicalEntityLoader:
    @staticmethod
    async def copy_all():
        """
        Copies all biomedical entities based on specs pulled from each document type

        NOTE: is slow due to UMLS linking (1-3 hours?)
        """
        patent_specs = PatentLoader.entity_specs()
        regulatory_approval_specs = RegulatoryApprovalLoader.entity_specs()
        trial_specs = TrialLoader.entity_specs()

        specs = patent_specs + regulatory_approval_specs + trial_specs

        for spec in specs:
            records = await PsqlDatabaseClient(spec.database).select(spec.sql)
            source_map = spec.get_source_map(records)
            terms = spec.get_terms(source_map)
            terms_to_canonicalize = spec.get_terms_to_canonicalize(source_map)
            await BiomedicalEntityEtl(
                candidate_selector=spec.candidate_selector,
                relation_id_field_map=spec.relation_id_field_map,
                non_canonical_source=spec.non_canonical_source,
            ).copy_all(terms, terms_to_canonicalize, source_map)

        await BiomedicalEntityEtl.pre_doc_finalize()


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
