"""
Entity client
"""
from pydash import compact, flatten
import logging

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def search(params: EntitySearchParams) -> list[Entity]:
    """
    Search for entities

    HACK: need to refactor data model such that this is shared between patents & trials
    """
    approvals = await approval_search_client(ApprovalSearchParams(**params.__dict__))
    patents = await patent_search_client(PatentSearchParams(**params.__dict__))
    trials = await trial_search_client(TrialSearchParams(**params.__dict__))

    if params.entity_types[0] != "pharmaceutical":
        raise NotImplementedError

    assets: set[str] = set(
        compact(
            flatten([p.interventions for p in patents])
            # + [t.instance_rollup for t in trials]
            # + [a.instance_rollup for a in approvals]
        )
    )

    logger.info("Got %s assets", len(assets))

    int_with_recs = [
        Entity(
            name=a,
            approvals=[app for app in approvals],  # if app.instance_rollup == a],
            patents=[p for p in patents if a in p.interventions],
            trials=[t for t in trials],  # if t.instance_rollup == a
        )
        for a in assets
    ]

    return sorted(int_with_recs, key=lambda e: e.record_count, reverse=True)
