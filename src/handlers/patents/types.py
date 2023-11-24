"""
Types used by patent handlers
"""
from pydantic import BaseModel

from clients.patents.types import QueryType, TermField


class BasePatentSearchParams(BaseModel):
    min_patent_years: int = 10
    limit: int = 800
    query_type: QueryType = "AND"


class OptionalRawPatentSearchParams(BasePatentSearchParams):
    exemplar_patents: str | None = None
    term_field: TermField = "terms"
    skip_cache: str | bool = False


class RawPatentSearchParams(OptionalRawPatentSearchParams):
    terms: str


class PatentSearchParams(BasePatentSearchParams):
    exemplar_patents: list[str] = []
    skip_cache: bool = False
    terms: list[str]
    term_field: TermField = "terms"
