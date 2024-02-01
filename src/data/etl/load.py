import asyncio
import sys

from .entity import BiomedicalEntityLoader, OwnerLoader, UmlsLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader


async def load_all(force_update: bool = False):
    """
    Central script for stage 2 of ETL (local dbs -> biosym)

    Args:
        force_update (bool, optional): Whether to update or merely create.
            If update, documents and their relations are first deleted.
    """
    # copy umls data
    await UmlsLoader().copy_all()

    # copy all biomedical entities (from all doc types)
    # Takes 3+ hours!!
    await BiomedicalEntityLoader().copy_all()

    # copy owner data (across all documents)
    await OwnerLoader().copy_all()

    # copy patent data
    await PatentLoader(document_type="patent").copy_all(force_update)

    # copy data about approvals
    await RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all(
        force_update
    )

    # copy trial data
    await TrialLoader(document_type="trial").copy_all(force_update)

    # do final biomedical entity stuff that requires everything else be in place
    await BiomedicalEntityLoader().post_doc_finalize()

    # finally, link owners
    await OwnerLoader().post_doc_finalize()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.load
            UMLS etl
        """
        )
        sys.exit()

    force_update = "--force_update" in sys.argv

    asyncio.run(load_all(force_update))
