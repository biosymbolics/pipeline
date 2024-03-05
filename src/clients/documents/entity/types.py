"""
Entity client
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field


from typings import ScoredPatent, ScoredRegulatoryApproval, ScoredTrial


@dataclass(frozen=True)
class DocsByType(BaseModel):
    patents: dict[str, ScoredPatent]
    regulatory_approvals: dict[str, ScoredRegulatoryApproval]
    trials: dict[str, ScoredTrial]


class DocResults(BaseModel):
    patents: list[str] = []
    regulatory_approvals: list[str] = []
    trials: list[str] = []


class EntWithDocResult(DocResults):
    # id: int
    name: str
    child: str | None = Field(validate_default=True)
