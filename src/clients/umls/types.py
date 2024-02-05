from pydantic import BaseModel
from prisma.enums import OntologyLevel


class NodeRecord(BaseModel):
    id: str
    name: str
    count: int | None = None
    level: OntologyLevel = OntologyLevel.UNKNOWN
    level_override: OntologyLevel | None = None
    type_ids: list[str] | None = None


class EdgeRecord(BaseModel):
    head: str
    tail: str
