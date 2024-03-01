"""
Vector client types
"""

from typing import Sequence

from pydantic import BaseModel, field_validator
from typings.core import ResultBase


class TopDocRecord(ResultBase):
    description: str
    id: str
    relevance_score: float
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
    avg_score: float
    scores: list[float]
    titles: list[str]
    descriptions: list[str] = []
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


MIN_YEAR = 2000
DEFAULT_K = 1000


class VectorSearchParams(BaseModel):
    min_year: int = MIN_YEAR
    skip_ids: Sequence[str] = []
    alpha: float = 0.7
    k: int = DEFAULT_K
    vector: list[float] = []

    def merge(self, new_params: dict) -> "VectorSearchParams":
        self_keep = {
            k: v for k, v in self.model_dump().items() if k not in new_params.keys()
        }
        return VectorSearchParams(
            **self_keep,
            **new_params,
        )
