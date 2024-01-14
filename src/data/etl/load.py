from .biomedical_entity import BiomedicalEntityLoader
from .documents import PatentLoader, RegulatoryApprovalLoader, TrialLoader
from .owner import OwnerLoader
from .umls import UmlsLoader


async def load_all():
    # copy umls data
    await UmlsLoader().copy_all()

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
