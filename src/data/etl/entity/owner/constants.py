from prisma.enums import OwnerType

from constants.company import COMPANY_STRINGS, LARGE_PHARMA_KEYWORDS
from core.ner.classifier import create_lookup_map

ASSIGNEE_PATENT_THRESHOLD = 20

OWNER_KEYWORD_MAP = create_lookup_map(
    {
        OwnerType.UNIVERSITY: [
            "univ(?:ersit(?:y|ies))?",
            "colleges?",
            "research hospitals?",
            "institute?s?",
            "schools?",
            "nyu",
            "universitaire?s?",
            # "l'Université",
            # "Université",
            "universita(?:ri)?",
            "education",
            "universidad",
        ],
        OwnerType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        OwnerType.INDUSTRY: [
            *COMPANY_STRINGS,
            "laboratories",
            "procter and gamble",
            "3m",
            "neuroscience$",
            "associates",
            "medical$",
        ],
        OwnerType.GOVERNMENTAL: [
            "government",
            "govt",
            "federal",
            "national",
            "state",
            "us health",
            "veterans affairs",
            "nih",
            "va",
            "european organisation",
            "eortc",
            "assistance publique",
            "fda",
            "bureau",
            "authority",
        ],
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
            "group$",
            "research cent(?:er|re)s?",
        ],
        OwnerType.INDIVIDUAL: [r"m\.?d\.?", "dr\\.?", "ph\\.?d\\.?"],
    }
)
