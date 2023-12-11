from typing import Literal

from pydantic import BaseModel


QueryType = Literal["AND", "OR"]


TermField = Literal["terms", "instance_rollup", "category_rollup"]


class BaseSearchParams(BaseModel):
    limit: int = 1000
    query_type: QueryType = "AND"
    skip_cache: str | bool = False


class BasePatentSearchParams(BaseSearchParams):
    min_patent_years: int = 10


class OptionalRawPatentSearchParams(BasePatentSearchParams):
    exemplar_patents: str | None = None
    term_field: TermField = "terms"


class RawPatentSearchParams(OptionalRawPatentSearchParams):
    terms: str


class PatentSearchParams(BasePatentSearchParams):
    exemplar_patents: list[str] = []
    terms: list[str]
    term_field: TermField = "terms"


class RawTrialSearchParams(BaseSearchParams):
    terms: str


class TrialSearchParams(BaseSearchParams):
    terms: list[str]
