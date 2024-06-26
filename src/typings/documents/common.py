from enum import Enum
import json
from typing import Any
from prisma.enums import BiomedicalEntityType
from pydantic import BaseModel, field_validator


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

    @staticmethod
    def from_doc_type(doc_type: DocType) -> "VectorizableRecordType":
        return {
            DocType.patent: VectorizableRecordType.patent,
            DocType.regulatory_approval: VectorizableRecordType.regulatory_approval,
            DocType.trial: VectorizableRecordType.trial,
        }[doc_type]


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
    name: str
    semantic_similarity: float
    syntactic_similarity: float
    synonyms: list[str]
    types: list[str]
    vector: list[float]

    @field_validator("vector", mode="before")
    def vector_from_string(cls, vec):
        # vec may be a string due to prisma limitations
        if isinstance(vec, str):
            return json.loads(vec)
        return vec
