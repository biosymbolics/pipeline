from pydantic import BaseModel
from prisma.enums import OntologyLevel


class NodeRecord(BaseModel):
    id: str
    count: int | None = None
    level: OntologyLevel = OntologyLevel.UNKNOWN


class EdgeRecord(BaseModel):
    head: str
    tail: str
