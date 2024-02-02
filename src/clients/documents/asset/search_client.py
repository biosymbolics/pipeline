"""
Asset client
"""

import asyncio
from functools import partial
import time
from typing import Sequence
from pydash import compact, flatten, group_by
import logging

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_client
from typings.client import (
    AssetSearchParams,
    DocumentSearchParams,
    PatentSearchParams as PatentParams,
    RegulatoryApprovalSearchParams as ApprovalParams,
    TrialSearchParams as TrialParams,
)
from typings.documents.common import EntityMapType, TermField
from typings.assets import Asset
from utils.string import get_id

from .. import approvals as regulatory_approval_client
from .. import patents as patent_client
from .. import trials as trial_client

from .types import DocsByType, EntWithDocResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_docs_by_entity_id(
    doc_ids: Sequence[str],
    rollup_field: TermField,
    child_field: TermField | None,
    entity_map_type: EntityMapType,
) -> list[EntWithDocResult]:
    """
    Gets entities for a set of doc ids
    """
    client = await prisma_client(120)

    query = f"""
        SELECT
            {rollup_field.name} as name,
            {f'{child_field.name} as child,' if child_field else ''}
            array_remove(array_agg(patent_id), NULL) as patents,
            array_remove(array_agg(regulatory_approval_id), NULL) as regulatory_approvals,
            array_remove(array_agg(trial_id), NULL) as trials
        FROM {entity_map_type.value} -- intervenable or indicatable
        WHERE (
            patent_id = ANY($1)
            OR regulatory_approval_id = ANY($1)
            OR trial_id = ANY($1)
        )
        AND entity_id is not NULL
        AND instance_rollup<>'' -- TODO?
        GROUP BY
            {rollup_field.name},
            {f'{child_field.name}' if child_field else ''}
        """
    result = await client.query_raw(query, doc_ids)

    ents_with_docs = [EntWithDocResult(**r) for r in result]

    return ents_with_docs


async def get_matching_docs(p: DocumentSearchParams) -> DocsByType:
    """
    Gets docs by type, matching doc_ids
    """

    regulatory_approvals = asyncio.create_task(
        regulatory_approval_client.search(ApprovalParams.parse(p, include=None))
    )
    patents = asyncio.create_task(
        patent_client.search(PatentParams.parse(p, include={"assignees": True}))
    )
    trials = asyncio.create_task(
        trial_client.search(TrialParams.parse(p, include={"sponsor": True}))
    )

    await asyncio.gather(regulatory_approvals, patents, trials)

    return DocsByType(
        regulatory_approvals={r.id: r for r in regulatory_approvals.result()},
        patents={p.id: p for p in patents.result()},
        trials={t.id: t for t in trials.result()},
    )


async def _search(p: DocumentSearchParams) -> list[Asset]:
    """
    Internal search for documents grouped by entity
    """
    start = time.monotonic()

    # full docs (if it were pulled in the prior query, would pull dups; thus like this.)
    docs_by_type = await get_matching_docs(p)

    doc_ids: list[str] = [k for d in flatten(docs_by_type.values()) for k in d.keys()]

    # entity/doc matching for ents in first order docs
    ent_with_docs = await get_docs_by_entity_id(
        doc_ids,
        TermField.instance_rollup,
        TermField.canonical_name,
        EntityMapType.intervention,
    )

    grouped_ents = group_by(ent_with_docs, lambda ewd: ewd.name)

    documents_by_entity = [
        Asset.create(
            id=rollup,
            name=ewds[0].name,
            children=[
                Asset.create(
                    id=rollup + ewd.child,
                    name=ewd.child,
                    children=[],
                    end_year=p.end_year,
                    patents=compact(
                        [docs_by_type.patents.get(id) for id in ewd.patents]
                    ),
                    regulatory_approvals=compact(
                        [
                            docs_by_type.regulatory_approvals.get(id)
                            for id in ewd.regulatory_approvals
                        ]
                    ),
                    start_year=p.start_year,
                    trials=compact([docs_by_type.trials.get(id) for id in ewd.trials]),
                )
                for ewd in ewds
                if ewd.child
            ],
            end_year=p.end_year,
            patents=compact(
                [
                    docs_by_type.patents.get(id)
                    for id in flatten([ewd.patents for ewd in ewds])
                ]
            ),
            regulatory_approvals=compact(
                [
                    docs_by_type.regulatory_approvals.get(id)
                    for id in flatten([ewd.regulatory_approvals for ewd in ewds])
                ]
            ),
            start_year=p.start_year,
            trials=compact(
                [
                    docs_by_type.trials.get(id)
                    for id in flatten([ewd.trials for ewd in ewds])
                ]
            ),
        )
        for rollup, ewds in grouped_ents.items()
    ]

    assets = sorted(documents_by_entity, key=lambda e: e.record_count, reverse=True)[
        0 : p.limit
    ]

    # half the time is in fetching all docs, the other half in transmitting.
    logger.info("Asset search took %s seconds", round(time.monotonic() - start))
    return assets


async def search(p: AssetSearchParams) -> list[Asset]:
    """
    Search for documents, grouped by entity
    """
    search_criteria = DocumentSearchParams.parse(p)
    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "assets",
        }
    )
    search_partial = partial(_search, search_criteria)

    # not caching for now; persist to s3 takes too long and it isn't worth.
    if True or p.skip_cache == True:
        assets = await search_partial()
        return assets

    return await retrieve_with_cache_check(
        search_partial,
        key=key,
        decode=lambda str_data: [Asset.load(**p) for p in storage_decoder(str_data)],
    )
