"""
Types used by patent handlers
"""
from typing import TypedDict
from typing_extensions import NotRequired


class BasePatentSearchParams(TypedDict):
    min_patent_years: NotRequired[int]
    max_results: NotRequired[int]
    skip_cache: NotRequired[bool]


class PatentSearchParams(BasePatentSearchParams):
    terms: str
    domains: NotRequired[str]


class ParsedPatentSearchParams(BasePatentSearchParams):
    terms: list[str]
    domains: NotRequired[list[str] | None]
