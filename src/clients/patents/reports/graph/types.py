from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    weight: int


@dataclass(frozen=True)
class Node:
    id: str
    group: str
    parent: str | None
    label: str
    size: int


@dataclass(frozen=True)
class SerializableGraph:
    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__)

    nodes: list[Node]
    edges: list[Link]
