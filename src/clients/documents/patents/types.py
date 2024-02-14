"""
Patent types
"""

from typing import Literal

from typings.core import ResultBase


RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]


class RelevanceByYear(ResultBase):
    year: int
    relevance: float


class BuyerRecord(ResultBase):
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
    max_relevance_score: float
    avg_relevance_score: float
    relevance_by_year: list[RelevanceByYear]
    score: float


class FindBuyerResult(ResultBase):
    buyers: list[BuyerRecord]
    description: str  # included since it could be a result of expansion
