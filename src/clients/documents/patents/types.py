"""
Patent types
"""

from typing import Literal

from typings.core import ResultBase


RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]


class CountByYear(ResultBase):
    year: int
    count: int


class CompanyRecord(ResultBase):
    id: int
    name: str
    ids: list[str]
    count: int
    symbol: str | None
    titles: list[str]
    # terms: list[str]
    min_age: int
    avg_age: float
    activity: list[float]
    avg_relevance_score: float
    wheelhouse_score: float
    count_by_year: list[CountByYear]
    score: float


class FindCompanyResult(ResultBase):
    companies: list[CompanyRecord]
    description: str  # included since it could be a result of expansion
