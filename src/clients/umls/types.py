from pydantic import BaseModel


class NodeRecord(BaseModel):
    id: str
    count: int


class EdgeRecord(BaseModel):
    head: str
    tail: str
