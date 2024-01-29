import asyncio
import sys

from .entity import BiomedicalEntityLoader, OwnerLoader, UmlsLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


async def load_all():
    # copy umls data
    # await UmlsLoader().copy_all()

    # # copy all biomedical entities (from all doc types)
    # await BiomedicalEntityLoader().copy_all()

    # # copy owner data (across all documents)
    # await OwnerLoader().copy_all()

    # copy patent data
    # await PatentLoader(document_type="patent").copy_all()

    # copy data about approvals
    # await RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all()

    # copy trial data
    # await TrialLoader(document_type="trial").copy_all()

    # do final biomedical entity stuff that requires everything else be in place
    await BiomedicalEntityLoader().post_doc_finalize()

    # finally, link owners
    # await OwnerLoader().post_doc_finalize()


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
