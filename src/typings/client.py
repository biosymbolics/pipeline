from datetime import datetime
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Discriminator, Field, Tag, field_validator
from prisma.types import PatentInclude, RegulatoryApprovalInclude, TrialInclude

from constants.patents import DEFAULT_PATENT_K
from typings.documents.common import DocType

from .documents.common import EntityCategory, TermField


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
            Annotated[None, Tag("none_include")],
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
    description: Annotated[str | None, Field(validate_default=True)] = None
    k: Annotated[int, Field(validate_default=True)] = DEFAULT_PATENT_K
    limit: Annotated[int, Field(validate_default=True)] = DEFAULT_LIMIT
    skip_cache: Annotated[bool, Field(validate_default=True)] = True


class PatentSearchParams(DocumentSearchParams):
    include: Annotated[PatentInclude | None, Field(validate_default=True)] = (
        DEFAULT_PATENT_INCLUDE
    )


class RegulatoryApprovalSearchParams(DocumentSearchParams):
    include: Annotated[
        Union[RegulatoryApprovalInclude, None], Field(validate_default=True)
    ] = DEFAULT_REGULATORY_APPROVAL_INCLUDE


class TrialSearchParams(DocumentSearchParams):
    include: Annotated[Union[TrialInclude, None], Field(validate_default=True)] = (
        DEFAULT_TRIAL_INCLUDE
    )


class EntitySearchParams(PatentSearchParams):
    # device, diagnostic, etc. not compound because it can be moa
    entity_category: Annotated[EntityCategory, Field(validate_default=True)] = (
        EntityCategory.intervention
    )
    include: Annotated[None | dict, Field()] = None

    @field_validator("entity_category", mode="before")
    def entity_category_from_string(cls, v):
        if isinstance(v, EntityCategory):
            return v
        return EntityCategory[v]


EntityField = Literal["interventions", "indications", "owners"]


class DocumentCharacteristicParams(DocumentSearchParams):
    """
    Parameters for document characteristics
    """

    doc_type: DocType = DocType.patent
    head_field: EntityField
    tail_field: EntityField
    include: Annotated[dict, Field()] = {}


class CompanyFinderParams(BaseModel):
    """
    Parameters for finding companies
    """

    description: Annotated[str | None, Field(validate_default=True)] = None
    similar_companies: Annotated[list[str], Field(validate_default=True)] = []
    k: Annotated[int, Field(validate_default=True)] = DEFAULT_PATENT_K
    min_year: Annotated[int, Field(validate_default=True)] = 2000
    use_gpt_expansion: Annotated[bool, Field(validate_default=True)] = False

    @field_validator("similar_companies", mode="before")
    def similar_companies_from_string(cls, v):
        if isinstance(v, list):
            return v
        similar_companies = [t.strip() for t in (v.split(";") if v else [])]
        return similar_companies


class ConceptDecomposeParams(BaseModel):
    """
    Parameters for concept decomposition handler
    """

    description: Annotated[str, Field(validate_default=True)]
    k: Annotated[int, Field(validate_default=True)] = DEFAULT_PATENT_K


AutocompleteType = Literal["entity", "owner"]


class AutocompleteParams(BaseModel):
    string: str
    limit: int = 25
    types: Annotated[list[AutocompleteType], Field(validate_default=True)] = [
        "entity",
        "owner",
    ]

    @field_validator("types", mode="before")
    def types_from_string(cls, v):
        if isinstance(v, list):
            return v
        types = [t.strip() for t in (v.split(";") if v else [])]
        return types
