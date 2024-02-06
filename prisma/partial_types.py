"""
Partial types https://prisma-client-py.readthedocs.io/en/stable/getting_started/partial-types/
Used for performance / avoiding big serializations
To apply changes: `prisma generate`
"""

from prisma.models import (
    Indicatable,
    Intervenable,
    Patent,
    Ownable,
    RegulatoryApproval,
    Trial,
)


Indicatable.create_partial(
    "IndicatableDto",
    exclude={
        "is_primary",
        "mention_index",
        "patent_id",
        "patent",
        "regulatory_approval",
        "regulatory_approval_id",
        "trial",
        "trial_id",
    },
)

Intervenable.create_partial(
    "IntervenableDto",
    exclude={
        "is_primary",
        "mention_index",
        "patent_id",
        "patent",
        "regulatory_approval",
        "regulatory_approval_id",
        "trial",
        "trial_id",
    },
)

Ownable.create_partial(
    "OwnableDto",
    exclude={
        "is_primary",
        "patent_id",
        "assignee_patent",
        "inventor_patent_id",
        "inventor_patent",
        "regulatory_approval",
        "regulatory_approval_id",
        "trial",
        "trial_id",
    },
)

Patent.create_partial(
    "PatentDto",
    exclude={"claims", "ipc_codes", "other_ids"},
    relations={
        "assignees": "OwnableDto",
        "interventions": "IntervenableDto",
        "indications": "IndicatableDto",
    },
)

RegulatoryApproval.create_partial(
    "RegulatoryApprovalDto",
    relations={
        "interventions": "IntervenableDto",
        "indications": "IndicatableDto",
    },
)

Trial.create_partial(
    "TrialDto",
    exclude={"abstract", "acronym", "arm_types"},
    relations={
        "sponsor": "OwnableDto",
        "interventions": "IntervenableDto",
        "indications": "IndicatableDto",
    },
)
