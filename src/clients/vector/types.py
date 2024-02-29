"""
Vector client types
"""

from typings.core import ResultBase


class TopDocRecord(ResultBase):
    id: str
    relevance_score: float
    title: str
    vector: list[float]
    year: int


class TopDocsByYear(ResultBase):
    ids: list[str]
    count: int
    avg_score: float
    scores: list[float]
    titles: list[str]
    total_score: float
    year: int


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
    relevance_score: float
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
