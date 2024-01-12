from enum import Enum
from prisma.enums import BiomedicalEntityType


class DocType(Enum):
    patent = "patent"
    regulatory_approval = "regulatory_approval"
    trial = "trial"


ENTITY_TYPE_MAP = BiomedicalEntityType.__members__
ENTITY_TYPES = ENTITY_TYPE_MAP.values()
ENTITY_DOMAINS = ENTITY_TYPES

DOMAINS_OF_INTEREST = [
    BiomedicalEntityType.MECHANISM.name,
    BiomedicalEntityType.COMPOUND.name,
    BiomedicalEntityType.BIOLOGIC.name,
    BiomedicalEntityType.DISEASE.name,
    # "assignees",
    # "inventors",
    # ATTRIBUTE_FIELD,
]
