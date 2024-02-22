"""
Vector client types
"""

from typings.core import ResultBase


class CountByYear(ResultBase):
    year: int
    count: int


class CompanyRecord(ResultBase):
    id: int
    name: str
    ids: list[str]
    is_acquirer: bool
    is_competition: bool
    count: int
    symbol: str | None
    titles: list[str]
    # terms: list[str]
    min_age: int
    avg_age: float
    activity: list[float]
    relevance_score: float
    wheelhouse_score: float
    count_by_year: list[CountByYear]
    score: float


class FindCompanyResult(ResultBase):
    companies: list[CompanyRecord]
    description: str  # included since it could be a result of expansion
    exit_score: float
    competition_score: float
