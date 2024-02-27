from enum import Enum
from prisma.enums import BiomedicalEntityType


class DocType(Enum):
    patent = "patent"
    regulatory_approval = "regulatory_approval"
    trial = "trial"


DOC_TYPE_DATE_MAP: dict[DocType, str] = {
    DocType.patent: "priority_date",
    DocType.regulatory_approval: "approval_date",
    DocType.trial: "start_date",
}


class TermField(Enum):
    canonical_name = "canonical_name"
    instance_rollup = "instance_rollup"
    category_rollup = "category_rollup"


class EntityCategory(Enum):
    intervention = "intervenable"
    indication = "indicatable"
    owner = "ownable"


ENTITY_MAP_TABLES = [t.value for t in EntityCategory]


ENTITY_TYPE_MAP = BiomedicalEntityType.__members__
ENTITY_TYPES = ENTITY_TYPE_MAP.values()
ENTITY_DOMAINS = ENTITY_TYPES

DOMAINS_OF_INTEREST = [
    BiomedicalEntityType.MECHANISM.name,
    BiomedicalEntityType.COMPOUND.name,
    BiomedicalEntityType.BIOLOGIC.name,
    BiomedicalEntityType.DISEASE.name,
    "OWNER",
    # "assignees",
    # "inventors",
    # ATTRIBUTE_FIELD,
]
