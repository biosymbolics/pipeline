from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator
from prisma.types import PatentInclude, RegulatoryApprovalInclude, TrialInclude

from .documents.common import EntityMapType, TermField


QueryType = Literal["AND", "OR"]
DEFAULT_QUERY_TYPE: QueryType = "AND"
DEFAULT_TERM_FIELDS = [
    TermField.canonical_name,
    TermField.instance_rollup,
]

DEFAULT_PATENT_INCLUDE: PatentInclude = {
    "assignees": True,
    # "inventors": True,
    "interventions": True,
    "indications": True,
}

DEFAULT_REGULATORY_APPROVAL_INCLUDE: RegulatoryApprovalInclude = {
    "interventions": True,
    "indications": True,
}

DEFAULT_TRIAL_INCLUDE: TrialInclude = {
    "interventions": True,
    "indications": True,
    "outcomes": True,
    "sponsor": True,
}


class CommonSearchParams(BaseModel):
    limit: Annotated[int, Field(validate_default=True)] = 1000
    query_type: Annotated[QueryType, Field(validate_default=True)] = "OR"  # TODO AND
    skip_cache: Annotated[bool, Field(validate_default=True)] = False
    terms: Annotated[list[str], Field(validate_default=True)] = []

    @field_validator("terms", mode="before")
    def terms_from_string(cls, v):
        if isinstance(v, list):
            return v
        terms = [t.strip() for t in (v.split(";") if v else [])]
        return terms


class PatentSearchParams(CommonSearchParams):
    exemplar_patents: Annotated[list[str], Field(validate_default=True)] = []
    include: PatentInclude = DEFAULT_PATENT_INCLUDE

    @field_validator("exemplar_patents", mode="before")
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class RegulatoryApprovalSearchParams(CommonSearchParams):
    include: RegulatoryApprovalInclude = DEFAULT_REGULATORY_APPROVAL_INCLUDE


class TrialSearchParams(CommonSearchParams):
    include: TrialInclude = DEFAULT_TRIAL_INCLUDE


GenericSearchParams = (
    PatentSearchParams | RegulatoryApprovalSearchParams | TrialSearchParams
)


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
