"""
Constants for the patents client.
"""

from .types import RelevancyThreshold


RELEVANCY_THRESHOLD_MAP: dict[RelevancyThreshold, float] = {
    "very low": 0.0,
    "low": 0.05,
    "medium": 0.20,
    "high": 0.50,
    "very high": 0.75,
}

MAX_PATENT_LIFE = 20

DOMAINS_OF_INTEREST = [
    "assignees",
    "compounds",
    "diseases",
    "genes",
    "inventors",
    "mechanisms",
]
