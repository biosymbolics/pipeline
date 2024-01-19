from datetime import datetime
from typing import Annotated, Literal, TypeVar
from pydantic import BaseModel, Field, field_validator
from prisma.types import PatentInclude, RegulatoryApprovalInclude, TrialInclude

from typings.documents.common import DocType

from .documents.common import EntityMapType, TermField


QueryType = Literal["AND", "OR"]
DEFAULT_QUERY_TYPE: QueryType = "AND"
DEFAULT_TERM_FIELDS = [
    TermField.canonical_name,
    TermField.instance_rollup,
]
DEFAULT_START_YEAR = datetime.today().year - 10
DEFAULT_END_YEAR = datetime.today().year + 1
DEFAULT_LIMIT = 2000

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


class DocumentSearchCriteria(BaseModel):
    """
    Document search criteria
    """

    end_year: Annotated[int, Field(validate_default=True)] = DEFAULT_END_YEAR
    query_type: Annotated[QueryType, Field(validate_default=True)] = DEFAULT_QUERY_TYPE
    start_year: Annotated[int, Field(validate_default=True)] = DEFAULT_START_YEAR
    terms: Annotated[list[str], Field(validate_default=True)] = []
    term_fields: Annotated[
        list[TermField], Field(validate_default=True)
    ] = DEFAULT_TERM_FIELDS

    @field_validator("terms", mode="before")
    def terms_from_string(cls, v):
        if isinstance(v, list):
            return v
        terms = [t.strip() for t in (v.split(";") if v else [])]
        return terms

    @classmethod
    def parse(cls, params: "DocumentSearchCriteria", **kwargs):
        p = {k: v for k, v in params.__dict__.items() if k in params.model_fields_set}
        return cls(**p, **kwargs)


class DocumentSearchParams(DocumentSearchCriteria):
    limit: Annotated[int, Field(validate_default=True)] = DEFAULT_LIMIT
    skip_cache: Annotated[bool, Field(validate_default=True)] = False


class PatentSearchParams(DocumentSearchParams):
    exemplar_patents: Annotated[list[str], Field(validate_default=True)] = []
    include: PatentInclude = DEFAULT_PATENT_INCLUDE

    @field_validator("exemplar_patents", mode="before")
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class RegulatoryApprovalSearchParams(DocumentSearchParams):
    include: RegulatoryApprovalInclude = DEFAULT_REGULATORY_APPROVAL_INCLUDE


class TrialSearchParams(DocumentSearchParams):
    include: TrialInclude = DEFAULT_TRIAL_INCLUDE


class AssetSearchParams(DocumentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_map_type: Annotated[
        EntityMapType, Field(validate_default=True)
    ] = EntityMapType.intervention

    @field_validator("entity_map_type", mode="before")
    def entity_map_type_from_string(cls, v):
        if isinstance(v, EntityMapType):
            return v
        return EntityMapType(v)


class DocumentCharacteristicParams(DocumentSearchParams):
    """
    Parameters for document characteristics
    """

    doc_type: DocType = DocType.patent
    head_field: str = "priority_date"
