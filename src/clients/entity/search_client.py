"""
Entity client
"""
import polars as pl
from pydash import flatten, uniq

from clients.patents import search as patent_search_client
from clients.trials import search as trial_search_client
from typings.client import EntitySearchParams, PatentSearchParams, TrialSearchParams
from typings.entities import Entity


def search(params: EntitySearchParams) -> list[Entity]:
    """
    Search for entities

    HACK: need to refactor data model such that this is shared between patents & trials
    """
    patents = patent_search_client(PatentSearchParams(**params.__dict__))
    trials = trial_search_client(TrialSearchParams(**params.__dict__))

    if params.entity_types[0] != "pharmaceutical":
        raise NotImplementedError

    interventions = uniq(
        flatten([p.interventions for p in patents] + [t.interventions for t in trials])
    )

    int_with_recs = [
        Entity(
            name=i,
            patents=[p for p in patents if i in p.interventions],
            trials=[t for t in trials if i in t.interventions],
        )
        for i in interventions
    ]

    return int_with_recs
