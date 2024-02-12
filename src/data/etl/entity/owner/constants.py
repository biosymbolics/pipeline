from prisma.enums import OwnerType

from constants.company import (
    COMMON_GOVT_WORDS,
    COMMON_UNIVERSITY_WORDS,
    COMMON_COMPANY_WORDS,
    LARGE_PHARMA_KEYWORDS,
)
from core.ner.classifier import create_lookup_map

ASSIGNEE_PATENT_THRESHOLD = 20


OWNER_KEYWORD_MAP = create_lookup_map(
    {
        OwnerType.UNIVERSITY: COMMON_UNIVERSITY_WORDS,
        OwnerType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        OwnerType.INDUSTRY: COMMON_COMPANY_WORDS,
        OwnerType.GOVERNMENTAL: COMMON_GOVT_WORDS,
        OwnerType.HEALTH_SYSTEM: [
            "healthcare",
            "(?:medical|cancer|health) (?:center|centre|system|hospital)s?",
            "clinics?",
            "districts?",
        ],
        OwnerType.FOUNDATION: ["foundatations?", "trusts?"],
        OwnerType.OTHER_ORGANIZATION: [
            "research network",
            "alliance",
            "research cent(?:er|re)s?",
        ],
        OwnerType.INDIVIDUAL: [r"m\.?d\.?", "dr\\.?", "ph\\.?d\\.?"],
    }
)
