"""
Entity client
"""
from pydash import flatten, uniq

from clients.approvals import search as approval_search_client
from clients.patents import search as patent_search_client
from clients.trials import search as trial_search_client
from typings.client import (
    ApprovalSearchParams,
    EntitySearchParams,
    PatentSearchParams,
    TrialSearchParams,
)
from typings.entities import Entity


def search(params: EntitySearchParams) -> list[Entity]:
    """
    Search for entities

    HACK: need to refactor data model such that this is shared between patents & trials
    """
    approvals = approval_search_client(ApprovalSearchParams(**params.__dict__))
    patents = patent_search_client(PatentSearchParams(**params.__dict__))
    trials = trial_search_client(TrialSearchParams(**params.__dict__))

    if params.entity_types[0] != "pharmaceutical":
        raise NotImplementedError

    interventions = uniq(
        flatten(
            [p.interventions for p in patents]
            + [t.interventions for t in trials]
            + [a.generic_name for a in approvals]
        )
    )

    int_with_recs = [
        Entity(
            name=i,
            approvals=[a for a in approvals if i == a.generic_name],
            patents=[p for p in patents if i in p.interventions],
            trials=[t for t in trials if i in t.interventions],
        )
        for i in interventions
    ]

    return sorted(int_with_recs, key=lambda e: e.record_count, reverse=True)
