from typing import Literal, Sequence

from pydantic import BaseModel


QueryType = Literal["AND", "OR"]


TermField = Literal["terms", "instance_rollup", "category_rollup"]


class BaseSearchParams(BaseModel):
    limit: int = 1000
    query_type: QueryType = "AND"
    skip_cache: str | bool = False


class CommonSearchParams(BaseSearchParams):
    terms: list[str]


class CommonRawSearchParams(BaseSearchParams):
    terms: str


class BasePatentSearchParams(CommonSearchParams):
    min_patent_years: int = 10


class OptionalRawPatentSearchParams(BasePatentSearchParams):
    exemplar_patents: str | None = None
    term_field: TermField = "terms"


class RawPatentSearchParams(CommonRawSearchParams):
    pass


class RawTrialSearchParams(CommonRawSearchParams):
    pass


class RawEntitySearchParams(CommonRawSearchParams):
    pass


class PatentSearchParams(BasePatentSearchParams):
    exemplar_patents: list[str] = []
    term_field: TermField = "terms"


class TrialSearchParams(CommonSearchParams):
    pass


class EntitySearchParams(CommonSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_types: Sequence[Literal["pharmaceutical"]] = ["pharmaceutical"]
