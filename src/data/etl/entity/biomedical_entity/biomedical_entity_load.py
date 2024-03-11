import asyncio
import sys
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from data.etl.documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


from .biomedical_entity import BiomedicalEntityEtl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiomedicalEntityLoader:
    @staticmethod
    async def copy_all(force_update: bool = False):
        """
        Copies all biomedical entities based on specs pulled from each document type

        NOTE: is slow due to UMLS linking (5-8 hours?)
        """
        if force_update:
            await BiomedicalEntityEtl.delete_all()

        patent_specs = PatentLoader.entity_specs()
        regulatory_approval_specs = RegulatoryApprovalLoader.entity_specs()
        trial_specs = TrialLoader.entity_specs()

        specs = patent_specs + regulatory_approval_specs + trial_specs

        for spec in specs:
            records = await PsqlDatabaseClient(spec.database).select(spec.sql)
            source_map = spec.get_source_map(records)
            terms = spec.get_terms(source_map)
            to_canonicalize = spec.get_terms_to_canonicalize(source_map)
            logger.info("ETLing %s terms", len(to_canonicalize[0]))
            bmee = await BiomedicalEntityEtl.create(
                candidate_selector_type=spec.candidate_selector,
                relation_id_field_map=spec.relation_id_field_map,
                non_canonical_source=spec.non_canonical_source,
            )
            await bmee.copy_all(terms, *to_canonicalize, source_map)

        logger.info("Biomedical entity load complete")

    @staticmethod
    async def finalize():
        await BiomedicalEntityEtl.finalize()
        return


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.biomedical_entity.biomedical_entity_load [--finalize] [--force_update]
            """
        )
        sys.exit()

    force_update = "--force_update" in sys.argv
    if "--finalize" in sys.argv:
        asyncio.run(BiomedicalEntityLoader().finalize())
    else:
        asyncio.run(BiomedicalEntityLoader().copy_all(force_update))
