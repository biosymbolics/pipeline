from prisma.enums import DropoutReason, TerminationReason

from nlp.classifier import create_lookup_map


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
            "no safety or efficacy concerns",
            "no safety concern",
        ],
        TerminationReason.NOT_FUTILITY: [
            "not related to efficacy",
            "no safety or efficacy concerns",
        ],
        TerminationReason.LOGISTICS: ["logistics", "logistical"],
    }
)


DROPOUT_REASON_KEYWORD_MAP = create_lookup_map(
    {
        DropoutReason.ADVERSE_EVENT: [
            "ae",
            "sae",
            "safety",
            "adverse",
            "toxicity",
            "abnormal laboratory",
            "laboratory abnormality",
            "hyperglycemia",
            "dlt",  # dose limiting toxicity
        ],
        DropoutReason.COMPLIANCE: [
            "compliance",
            "noncompliance",
            "excluded medication",
            "prohibited",  # prohibited meds
            "positive drug screen",
        ],
        DropoutReason.MILD_ADVERSE_EVENT: ["discomfort", "mild adverse event"],
        DropoutReason.DEATH: [
            "death",
        ],
        DropoutReason.INELIGIBLE: [
            "ineligible",
            "not treated",  # ??
            "screen failure",
            "entry criteria",
            "entrance criteria",
            "no longer meets",  # post hoc
            "eligibility",
            "ineligibility",
            "inappropriate enrollment",  # post hoc
            "selection criteria",  # pre or post hoc?
        ],
        DropoutReason.EFFICACY: [
            "efficacy",  # Loss of efficacy
            "unsatisfactory therapeutic effect",
            "insufficient clinical response",
            "treatment failure",
            "virologic failure",
            "did not achieve",
            "non-responder",
        ],
        DropoutReason.INVESTIGATOR: [
            "investigator",  # "Investigator's Discretion"
            "sponsor",  # ??
        ],
        DropoutReason.LOGISTICS: [
            "administrative problems",
            "administrative",
            "technical",  # Technical problems
            "site closed",
            "logistical",
        ],
        DropoutReason.LOST_TO_FOLLOWUP: [
            "lost to follow up",
            "missing",
            "missed",
            "failure to return",
        ],
        DropoutReason.OTHER: [
            "miscellaneous",
            "other",
            "not specified",
            "not reported",
            "unclassified",
            "unable to classify",
        ],
        DropoutReason.PHYSICIAN: [
            "physician",
            "alternative treatment",
        ],
        DropoutReason.PREGNANCY: [
            "pregnancy",
        ],
        DropoutReason.PROGRESSION: [
            "progression",
            "progressive",
            "deterioration",
            "relapse",
            "condition under investigation worsened",
            "debilitation",
            "graft loss",  # ??
            "hospitalization",
            "invasive intervention",  # ??
        ],
        DropoutReason.PROTOCOL_VIOLATION: [
            "protocol violation",
            "protocol deviation",
        ],
        DropoutReason.PER_PROTOCOL: [
            "stopping criteria",
            "withdrawal criteria",
            "early termination",
            "per protocol",
        ],
        DropoutReason.SUBJECT: [
            "subject",
            "consent",
            "no longer willing",  # No longer willing to participate
            "voluntary",
            "moved",  # moved from area
            "parent",  # Withdrawal by Parent/Guardian
            "personal",
            "relocation",
            "refused",
            "participant request",
            "incarcerated",
            "travel",
            "caregiver",
            "family emergency",
            "inconvenience",
        ],
        DropoutReason.TERMINATION: [
            "termination",
            "terminated",
            "site closure",
        ],
    }
)
