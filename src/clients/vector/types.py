"""
Vector client types
"""

from pydantic import field_validator
from typings.core import ResultBase


class TopDocRecord(ResultBase):
    description: str
    id: str
    relevance: float
    title: str
    vector: list[float]
    year: int

    @field_validator("vector", mode="before")
    def vector_from_string(cls, vec):
        # vec may be a string due to prisma limitations
        if isinstance(vec, str):
            return [float(v) for v in vec.strip("[]").split(",")]
        return vec


class TopDocsByYear(ResultBase):
    ids: list[str]
    count: int
    descriptions: list[str] = []
    titles: list[str]
    year: int

    avg_relevance: float
    total_investment: float
    total_relevance: float
    total_traction: float


class CountByYear(ResultBase):
    year: int
    count: int
    type: str


class UrlDef(ResultBase):
    title: str
    url: str


class CompanyRecord(ResultBase):
    id: int
    name: str
    ids: list[str]
    is_acquirer: bool
    is_competition: bool
    count: int
    symbol: str | None
    titles: list[str]
    urls: list[UrlDef]
    # terms: list[str]
    min_age: int
    avg_age: float
    activity: list[float] = []
    relevance: float
    wheelhouse_score: float = 0.0
    count_by_year: dict[str, list[CountByYear]] = {}
    score: float


class FindCompanyResult(ResultBase):
    companies: list[CompanyRecord]
    description: str  # included since it could be a result of expansion
    exit_score: float
    competition_score: float


class SubConcept(ResultBase):
    name: str
    description: str
    report: list[TopDocsByYear] = []
