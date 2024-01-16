import asyncio
import sys

from clients.low_level.prisma import prisma_context
from .entity import BiomedicalEntityLoader, OwnerLoader, UmlsLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


async def load_all():
    async with prisma_context(300):
        # copy umls data
        # await UmlsLoader().copy_all()

        # copy all biomedical entities (from all doc types)
        await BiomedicalEntityLoader().copy_all()

        # copy owner data (across all documents)
        await OwnerLoader().copy_all()

        # copy patent data
        await PatentLoader(document_type="patent").copy_all()

        # copy data about approvals
        await RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all()

        # copy trial data
        await TrialLoader(document_type="trial").copy_all()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.load
            UMLS etl
        """
        )
        sys.exit()

    asyncio.run(load_all())
