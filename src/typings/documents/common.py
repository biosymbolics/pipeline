from enum import Enum
from prisma.enums import BiomedicalEntityType
from pydantic import BaseModel


class DocType(Enum):
    patent = "patent"
    regulatory_approval = "regulatory_approval"
    trial = "trial"
    all = "all"


class VectorizableRecordType(Enum):
    patent = "patent"
    regulatory_approval = "regulatory_approval"
    trial = "trial"
    umls = "umls"


DOC_TYPE_DATE_MAP: dict[DocType, str] = {
    DocType.patent: "priority_date",
    DocType.regulatory_approval: "approval_date",
    DocType.trial: "start_date",
}

DOC_TYPE_DEDUP_ID_MAP: dict[DocType, str] = {
    DocType.patent: "family_id",
    DocType.regulatory_approval: "id",
    DocType.trial: "id",
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


class MentionCandidate(BaseModel):
    id: str
    # definition: str
    name: str
    similarity: float
    synonyms: list[str]
    types: list[str]
