from prisma.models import RegulatoryApproval

from typings.core import EntityBase


class ScoredRegulatoryApproval(RegulatoryApproval, EntityBase):
    pass
