"""
Partial types https://prisma-client-py.readthedocs.io/en/stable/getting_started/partial-types/
Used for performance / avoiding big serializations
To apply changes: `prisma generate`
"""

from prisma.models import (
    Indicatable,
    Intervenable,
    MockChat,
    Patent,
    Ownable,
    RegulatoryApproval,
    Trial,
)


Indicatable.create_partial(
    "IndicatableDto",
    exclude={
        "id",
        # "canonical_name", # used in doc modals
        "entity",
        "entity_id",
        "instance_rollup",
        "category_rollup",
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
        "id",
        # "canonical_name", # used in doc modals
        "entity",
        "entity_id",
        "instance_rollup",
        "category_rollup",
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
        "id",
        # "canonical_name", # used in assets
        "instance_rollup",
        "category_rollup",
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
    exclude={"application_number", "claims", "country_code", "ipc_codes", "other_ids"},
    relations={
        "assignees": "OwnableDto",
        "interventions": "IntervenableDto",
        "indications": "IndicatableDto",
    },
)

RegulatoryApproval.create_partial(
    "RegulatoryApprovalDto",
    relations={
        "applicant": "OwnableDto",
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


MockChat.create_partial(
    "Chat",
    exclude={"id"},
)
