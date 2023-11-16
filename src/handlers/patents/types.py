"""
Types used by patent handlers
"""
from typing import TypedDict
from typing_extensions import NotRequired

from clients.patents.types import QueryType, TermField


class BasePatentSearchParams(TypedDict):
    min_patent_years: NotRequired[int]
    limit: NotRequired[int]
    query_type: NotRequired[QueryType]


class OptionalPatentSearchParams(BasePatentSearchParams):
    is_exhaustive: NotRequired[str | bool]
    term_field: NotRequired[TermField]
    skip_cache: NotRequired[str | bool]


class PatentSearchParams(OptionalPatentSearchParams):
    terms: str


class ParsedPatentSearchParams(BasePatentSearchParams):
    is_exhaustive: NotRequired[bool]
    skip_cache: NotRequired[bool]
    terms: list[str]
    term_field: NotRequired[TermField]
