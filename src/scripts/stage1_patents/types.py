from typing import Optional, TypedDict


class BaseTermRecord(TypedDict):
    term: str
    count: int
    id: Optional[str]
    ids: Optional[list[str]]


class TermRecord(BaseTermRecord):
    domain: str
    original_term: Optional[str]


class AggregatedTermRecord(BaseTermRecord):
    domains: list[str]
    synonyms: list[str]
