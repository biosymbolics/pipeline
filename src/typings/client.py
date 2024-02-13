from datetime import datetime
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Discriminator, Field, Tag, field_validator
from prisma.types import PatentInclude, RegulatoryApprovalInclude, TrialInclude

from constants.patents import DEFAULT_BUYER_K
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
    "applicant": True,
    "interventions": True,
    "indications": True,
}
DEFAULT_TRIAL_INCLUDE: TrialInclude = {
    "dropout_reasons": True,
    "interventions": True,
    "indications": True,
    "outcomes": True,
    "sponsor": True,
}


def include_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        if v.get("assignees"):
            return "patent_include"
        elif v.get("sponsor"):
            return "trial_include"
        elif v.get("interventions"):
            return "regulatory_approval_include"

    return "no_include"


class TermSearchCriteria(BaseModel):
    query_type: Annotated[QueryType, Field(validate_default=True)] = DEFAULT_QUERY_TYPE
    terms: Annotated[list[str], Field(validate_default=True)] = []
    term_fields: Annotated[list[TermField], Field(validate_default=True)] = (
        DEFAULT_TERM_FIELDS
    )

    @field_validator("term_fields", mode="before")
    def term_fields_from_string(cls, v):
        if isinstance(v, list):
            return v
        term_fields = [t.strip() for t in (v.split(";") if v else [])]
        return term_fields


class DocumentSearchCriteria(TermSearchCriteria):
    """
    Document search criteria
    """

    end_year: Annotated[int, Field(validate_default=True)] = DEFAULT_END_YEAR
    # TODO: HACK!!
    include: Annotated[
        Union[
            Annotated[PatentInclude, Tag("patent_include")],
            Annotated[RegulatoryApprovalInclude, Tag("regulatory_approval_include")],
            Annotated[TrialInclude, Tag("trial_include")],
            Annotated[dict, Tag("no_include")],
        ],
        Discriminator(include_discriminator),
    ]
    start_year: Annotated[int, Field(validate_default=True)] = DEFAULT_START_YEAR

    @field_validator("terms", mode="before")
    def terms_from_string(cls, v):
        if isinstance(v, list):
            return v
        terms = [t.strip() for t in (v.split(";") if v else [])]
        return terms

    @field_validator("include", mode="before")
    def include_from_string(cls, i):
        if isinstance(i, dict):
            return i
        includes = [t.strip() for t in (i.split(";") if i else [])]
        return {i: True for i in includes}

    @classmethod
    def parse(cls, params: "DocumentSearchCriteria", **kwargs):
        p = {k: v for k, v in params.__dict__.items() if k in cls.model_fields.keys()}
        return cls(**{**p, **kwargs})


class DocumentSearchParams(DocumentSearchCriteria):
    limit: Annotated[int, Field(validate_default=True)] = DEFAULT_LIMIT
    skip_cache: Annotated[bool, Field(validate_default=True)] = True


class PatentSearchParams(DocumentSearchParams):
    exemplar_patents: Annotated[list[str], Field(validate_default=True)] = []
    include: Annotated[Union[PatentInclude, dict], Field(validate_default=True)] = (
        DEFAULT_PATENT_INCLUDE
    )

    @field_validator("exemplar_patents", mode="before")
    def exemplar_patents_from_string(cls, v):
        if isinstance(v, list):
            return v
        patents = [t.strip() for t in (v.split(";") if v else [])]
        return patents


class RegulatoryApprovalSearchParams(DocumentSearchParams):
    include: Annotated[
        Union[RegulatoryApprovalInclude, dict], Field(validate_default=True)
    ] = DEFAULT_REGULATORY_APPROVAL_INCLUDE


class TrialSearchParams(DocumentSearchParams):
    include: Annotated[Union[TrialInclude, dict], Field(validate_default=True)] = (
        DEFAULT_TRIAL_INCLUDE
    )


class AssetSearchParams(DocumentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_map_type: Annotated[EntityMapType, Field(validate_default=True)] = (
        EntityMapType.intervention
    )
    include: Annotated[dict, Field()] = {}

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
    include: Annotated[dict, Field()] = {}


class BuyerFinderParams(BaseModel):
    """
    Parameters for finding potential buyers
    """

    description: Annotated[str, Field(validate_default=True)]
    k: Annotated[int, Field(validate_default=True)] = DEFAULT_BUYER_K
    use_gpt_expansion: Annotated[bool, Field(validate_default=True)] = False
