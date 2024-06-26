from prisma.enums import OwnerType

from constants.company import (
    COMMON_FOUNDATION_WORDS,
    COMMON_GOVT_WORDS,
    COMMON_HEALTH_SYSTEM_WORDS,
    COMMON_INDIVIDUAL_WORDS,
    COMMON_UNIVERSITY_WORDS,
    COMMON_COMPANY_WORDS,
    LARGE_PHARMA_KEYWORDS,
    COMMON_NON_PROFIT_WORDS,
)
from nlp.classifier import create_lookup_map


OwnerTypePriorityMap = {
    OwnerType.INDUSTRY_LARGE: 1,
    OwnerType.HEALTH_SYSTEM: 5,
    OwnerType.INDIVIDUAL: 7,  # uncommon match
    OwnerType.UNIVERSITY: 10,
    OwnerType.GOVERNMENTAL: 20,
    OwnerType.FOUNDATION: 30,
    OwnerType.NON_PROFIT: 40,
    OwnerType.INDUSTRY: 50,
    OwnerType.OTHER_ORGANIZATION: 100,
    OwnerType.OTHER: 1000,
}


OWNER_KEYWORD_MAP = create_lookup_map(
    {
        OwnerType.UNIVERSITY: COMMON_UNIVERSITY_WORDS,
        OwnerType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        OwnerType.INDUSTRY: COMMON_COMPANY_WORDS,
        OwnerType.GOVERNMENTAL: COMMON_GOVT_WORDS,
        OwnerType.HEALTH_SYSTEM: COMMON_HEALTH_SYSTEM_WORDS,
        OwnerType.FOUNDATION: COMMON_FOUNDATION_WORDS,
        OwnerType.NON_PROFIT: [
            "research network",
            "research cent(?:er|re)s?",
            *COMMON_NON_PROFIT_WORDS,
        ],
        OwnerType.OTHER_ORGANIZATION: [],
        OwnerType.INDIVIDUAL: COMMON_INDIVIDUAL_WORDS,
    }
)
