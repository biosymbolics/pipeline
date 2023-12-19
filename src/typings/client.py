from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, Field, validator


QueryType = Literal["AND", "OR"]


TermField = Literal["terms", "instance_rollup", "category_rollup"]


class BaseSearchParams(BaseModel):
    limit: Annotated[int, Field(validate_default=True)] = 1000
    query_type: Annotated[QueryType, Field(validate_default=True)] = "AND"
    skip_cache: Annotated[bool, Field(validate_default=True)] = False
    term_field: Annotated[TermField, Field(validate_default=True)] = "terms"


class BasePatentSearchParams(BaseSearchParams):
    min_patent_years: Annotated[int, Field(validate_default=True)] = 10


class CommonSearchParams(BaseSearchParams):
    terms: Annotated[list[str], Field(validate_default=True)] = []

    @validator("terms", pre=True)
    def terms_from_string(cls, v):
        if isinstance(v, list):
            return v
        terms = [t.strip() for t in (v.split(";") if v else [])]
        return terms


class PatentSearchParams(BasePatentSearchParams, CommonSearchParams):
    exemplar_patents: Annotated[list[str], Field(validate_default=True)] = []

    @validator("exemplar_patents", pre=True)
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class TrialSearchParams(CommonSearchParams):
    pass


class EntitySearchParams(PatentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_types: Annotated[
        Sequence[Literal["pharmaceutical"]], Field(validate_default=True)
    ] = ["pharmaceutical"]

    @validator("entity_types", pre=True)
    def entity_types_from_string(cls, v):
        if isinstance(v, list):
            return v
        entity_types = [t.strip() for t in (v.split(";") if v else [])]
        return entity_types


class ApprovalSearchParams(CommonSearchParams):
    pass
