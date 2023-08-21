"""
Types used by patent handlers
"""
from typing import TypedDict
from typing_extensions import NotRequired

from clients.patents import RelevancyThreshold


class PatentSearchParams(TypedDict):
    terms: str
    fetch_approval: NotRequired[bool]
    min_patent_years: NotRequired[int]
    relevancy_threshold: NotRequired[RelevancyThreshold]
    max_results: NotRequired[int]
