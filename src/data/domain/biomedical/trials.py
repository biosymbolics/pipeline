from prisma.enums import TerminationReason

from core.ner.classifier import create_lookup_map


TERMINATION_KEYWORD_MAP = create_lookup_map(
    {
        TerminationReason.FUTILITY: [
            "futility",
            # "efficacy",
            "endpoints?",  # "failed to meet (?:primary )?endpoint", "no significant difference on (?:primary )?endpoint", "efficacy endpoints?", "efficacy endpoints"
            "lower success rates?",
            "interim analysis",
            "lack of effectiveness",
            "lack of efficacy",
            "rate of relapse",
            "lack of response",
            "lack of performance",
            "inadequate effect",
            "no survival benefit",
            "stopping rule",
        ],
        TerminationReason.SAFETY: [
            # "safety", # "not a safety issue"
            "toxicity",
            "adverse",
            "risk/benefit",
            "detrimental effect",
            "S?AEs?",
            "mortality",
            # "safety concerns?",
            "unacceptable morbidity",
            "side effects?",
            "lost to follow up",  # TODO: is this really a safety issue?
        ],
        TerminationReason.BUSINESS: [
            "business",
            "company",
            "strategic",
            "sponsor(?:'s)? decision",
            "management",
            "stakeholders?",
            "(?:re)?prioritization",
        ],
        TerminationReason.ENROLLMENT: [
            "accruals?",
            "enroll?(?:ment|ed)?",
            "inclusions?",
            "recruit(?:ment|ing)?s?",
            "lack of (?:eligibile )?(?:participants?|subjects?|patients?)",
        ],
        TerminationReason.INVESTIGATOR: ["investigator", "PI"],
        TerminationReason.FUNDING: ["funding", "resources", "budget", "financial"],
        TerminationReason.COVID: ["covid", "coronavirus", "pandemic"],
        TerminationReason.OVERSIGHT: [
            "IRB",
            "ethics",  # "ethics committee",
            "Institutional Review Board",
            "certification",
            "FDA",
        ],
        TerminationReason.SUPPLY_CHAIN: [
            "supply",
            "unavalaible",
            "shortage",
            "manufacturing",
        ],
        TerminationReason.PROTOCOL_REVISION: [
            "revision",
            "change in (?:study )?protocol",
        ],
        TerminationReason.FEASIBILITY: ["feasibility"],
        TerminationReason.NOT_SAFETY: [
            "not a safety issue",
            "not related to safety",
            "No Safety or Efficacy Concerns",
            "no safety concern",
        ],
        TerminationReason.NOT_FUTILITY: [
            "not related to efficacy",
            "No Safety or Efficacy Concerns",
        ],
        TerminationReason.LOGISTICS: ["logistics", "logistical"],
    }
)
