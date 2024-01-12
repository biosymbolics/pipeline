from dataclasses import dataclass
import json
from typing import Literal

from typings.core import Dataclass


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


@dataclass(frozen=True)
class AggregateDocumentRelationship(Dataclass):
    head: str
    concept: str
    count: int
    documents: list[str]
