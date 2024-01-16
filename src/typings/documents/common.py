from enum import Enum
from prisma.enums import BiomedicalEntityType


class DocType(Enum):
    patent = "patent"
    regulatory_approval = "regulatory_approval"
    trial = "trial"


class TermField(Enum):
    canonical_name = "canonical_name"
    instance_rollup = "instance_rollup"
    category_rollup = "category_rollup"


# TODO: arcane name
class EntityMapType(Enum):
    intervention = "intervenable"
    indication = "indicatable"


ENTITY_MAP_TABLES = [t.value for t in EntityMapType]


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
