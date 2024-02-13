"""
Constants for the patents client.
"""

from .types import RelevancyThreshold


DEFAULT_BUYER_K = 1000

RELEVANCY_THRESHOLD_MAP: dict[RelevancyThreshold, float] = {
    "very low": 0.0,
    "low": 0.05,
    "medium": 0.20,
    "high": 0.50,
    "very high": 0.75,
}

EST_MAX_CLINDEV = 10
MAX_PATENT_LIFE = 20
