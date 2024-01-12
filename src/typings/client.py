from typing import Annotated, Literal
from prisma.types import PatentInclude
from pydantic import BaseModel, Field, field_validator

from typings.documents.common import EntityMapType, TermField


QueryType = Literal["AND", "OR"]


class BaseSearchParams(BaseModel):
    limit: Annotated[int, Field(validate_default=True)] = 1000
    query_type: Annotated[QueryType, Field(validate_default=True)] = "OR"  # TODO AND
    skip_cache: Annotated[bool, Field(validate_default=True)] = False
    term_field: Annotated[
        TermField, Field(validate_default=True)
    ] = TermField.canonical_name


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
    # None will be replaced with default (all relations)
    include: PatentInclude | None = None

    @field_validator("exemplar_patents", mode="before")
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class TrialSearchParams(CommonSearchParams):
    pass


class AssetSearchParams(PatentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_map_type: Annotated[
        EntityMapType, Field(validate_default=True)
    ] = EntityMapType.intervention

    @field_validator("entity_map_type", mode="before")
    def entity_map_type_from_string(cls, v):
        if isinstance(v, EntityMapType):
            return v
        return EntityMapType(v)


class ApprovalSearchParams(CommonSearchParams):
    pass
