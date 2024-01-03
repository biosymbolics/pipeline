"""
Constants for the patents client.
"""
from prisma.enums import BiomedicalEntityType

from constants.patents import ATTRIBUTE_FIELD
from .types import RelevancyThreshold


RELEVANCY_THRESHOLD_MAP: dict[RelevancyThreshold, float] = {
    "very low": 0.0,
    "low": 0.05,
    "medium": 0.20,
    "high": 0.50,
    "very high": 0.75,
}

EST_MAX_CLINDEV = 10
MAX_PATENT_LIFE = 20


ENTITY_TYPE_MAP = BiomedicalEntityType.__members__
ENTITY_TYPES = ENTITY_TYPE_MAP.values()
ENTITY_DOMAINS = ENTITY_TYPES

DOMAINS_OF_INTEREST = [
    *ENTITY_DOMAINS,
    "assignees",
    ATTRIBUTE_FIELD,
    "inventors",
]
