from typing import Literal, Sequence

from pydantic import BaseModel


QueryType = Literal["AND", "OR"]


TermField = Literal["terms", "instance_rollup", "category_rollup"]


class BaseSearchParams(BaseModel):
    limit: int = 1000
    query_type: QueryType = "AND"
    skip_cache: str | bool = False


class CommonRawSearchParams(BaseSearchParams):
    terms: str


class BasePatentSearchParams(BaseSearchParams):
    min_patent_years: int = 10
    term_field: TermField = "terms"


class OptionalPatentSearchParams(BasePatentSearchParams):
    exemplar_patents: list[str] = []


class RawPatentSearchParams(BasePatentSearchParams, CommonRawSearchParams):
    exemplar_patents: str | None = None


class RawTrialSearchParams(CommonRawSearchParams):
    pass


class RawEntitySearchParams(CommonRawSearchParams):
    pass


class CommonSearchParams(BaseSearchParams):
    terms: list[str]


class PatentSearchParams(BasePatentSearchParams, CommonSearchParams):
    exemplar_patents: list[str] = []


class TrialSearchParams(CommonSearchParams):
    pass


class EntitySearchParams(CommonSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_types: Sequence[Literal["pharmaceutical"]] = ["pharmaceutical"]
