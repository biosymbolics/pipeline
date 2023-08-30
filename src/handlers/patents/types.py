"""
Types used by patent handlers
"""
from typing import TypedDict
from typing_extensions import NotRequired


class BasePatentSearchParams(TypedDict):
    min_patent_years: NotRequired[int]
    limit: NotRequired[int]


class PatentSearchParams(BasePatentSearchParams):
    is_exhaustive: NotRequired[str | bool]
    terms: str
    domains: NotRequired[str]
    skip_cache: NotRequired[str | bool]


class ParsedPatentSearchParams(BasePatentSearchParams):
    is_exhaustive: NotRequired[bool]
    skip_cache: NotRequired[bool]
    terms: list[str]
    domains: NotRequired[list[str] | None]
