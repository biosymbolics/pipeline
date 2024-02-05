from typing import Union
from prisma.models import (
    BiomedicalEntity,
    Patent,
    RegulatoryApproval,
    Owner,
    Trial,
    TrialOutcome,
    Umls,
)
from prisma.types import (
    BiomedicalEntityCreateInput,
    BiomedicalEntityUpdateInput,
    BiomedicalEntityCreateWithoutRelationsInput,
    OwnerCreateWithoutRelationsInput,
    UmlsUpdateInput,
)


class BiomedicalEntityCreateInputWithRelationIds(
    BiomedicalEntityCreateWithoutRelationsInput
):
    comprised_of: list[str]
    parents: list[str]
    synonyms: list[str]


class OwnerCreateWithSynonymsInput(OwnerCreateWithoutRelationsInput):
    synonyms: list[str]


# Add more as needed
# (or create a proper base class that works for clients.low_level.prisma.batch_update)
AllModelTypes = Union[
    dict,
    BiomedicalEntity,
    BiomedicalEntityCreateInput,
    BiomedicalEntityUpdateInput,
    BiomedicalEntityCreateInputWithRelationIds,
    Patent,
    RegulatoryApproval,
    Owner,
    OwnerCreateWithSynonymsInput,
    Trial,
    TrialOutcome,
    Umls,
    UmlsUpdateInput,
]
