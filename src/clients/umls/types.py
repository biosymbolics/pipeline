from pydantic import BaseModel
from prisma.enums import OntologyLevel


class NodeRecord(BaseModel):
    id: str
    count: int | None = None
    level: OntologyLevel = OntologyLevel.UNKNOWN
    level_override: OntologyLevel | None = None


class EdgeRecord(BaseModel):
    head: str
    tail: str
