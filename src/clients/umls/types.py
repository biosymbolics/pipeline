from pydantic import BaseModel
from prisma.enums import OntologyLevel


class NodeRecord(BaseModel):
    id: str
    count: int
    level: OntologyLevel


class EdgeRecord(BaseModel):
    head: str
    tail: str
