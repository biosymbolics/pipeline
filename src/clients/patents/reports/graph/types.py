from dataclasses import dataclass


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    weight: int


@dataclass(frozen=True)
class Node:
    id: str
    patent_ids: list[str]
    size: int


@dataclass(frozen=True)
class SerializableGraph:
    nodes: list[Node]
    links: list[Link]
