from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, Field, field_validator


QueryType = Literal["AND", "OR"]
TermField = Literal["name", "canonical_name", "instance_rollup"]


class BaseSearchParams(BaseModel):
    limit: Annotated[int, Field(validate_default=True)] = 1000
    query_type: Annotated[QueryType, Field(validate_default=True)] = "OR"  # TODO AND
    skip_cache: Annotated[bool, Field(validate_default=True)] = False
    term_field: Annotated[TermField, Field(validate_default=True)] = "canonical_name"


class BasePatentSearchParams(BaseSearchParams):
    min_patent_years: Annotated[int, Field(validate_default=True)] = 10


class CommonSearchParams(BaseSearchParams):
    terms: Annotated[list[str], Field(validate_default=True)] = []

    @field_validator("terms", mode="before")
    def terms_from_string(cls, v):
        if isinstance(v, list):
            return v
        terms = [t.strip() for t in (v.split(";") if v else [])]
        return terms


class PatentSearchParams(BasePatentSearchParams, CommonSearchParams):
    exemplar_patents: Annotated[list[str], Field(validate_default=True)] = []

    @field_validator("exemplar_patents", mode="before")
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class TrialSearchParams(CommonSearchParams):
    pass


EntityType = Literal["intervention", "indication"]


class AssetSearchParams(PatentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_types: Annotated[Sequence[EntityType], Field(validate_default=True)] = [
        "intervention"
    ]

    @field_validator("entity_types", mode="before")
    def entity_types_from_string(cls, v):
        if isinstance(v, list):
            return v
        entity_types = [t.strip() for t in (v.split(";") if v else [])]
        return entity_types


class ApprovalSearchParams(CommonSearchParams):
    pass
