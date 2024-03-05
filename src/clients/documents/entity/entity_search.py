"""
Entity client
"""

import asyncio
from functools import partial
import time
from typing import Mapping
from pydash import compact, flatten, group_by
import logging

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.prisma import prisma_context
from typings.client import (
    EntitySearchParams,
    DocumentSearchParams,
    PatentSearchParams as PatentParams,
    RegulatoryApprovalSearchParams as ApprovalParams,
    TrialSearchParams as TrialParams,
)
from typings.documents.common import DocType, EntityCategory, TermField
from typings.entities import Entity
from utils.string import get_id

from .. import approvals as regulatory_approval_client
from .. import patents as patent_client
from .. import trials as trial_client

from .types import DocsByType, EntWithDocResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_docs_by_entity_id(
    doc_id_map: Mapping[str, list[str]],
    rollup_field: TermField,
    child_field: TermField | None,
    entity_category: EntityCategory,
) -> list[EntWithDocResult]:
    """
    Gets entities for a set of doc ids
    Returns entity info with list of the various document ids (patent/regulatory_approval/trial)
    """

    query = f"""
        SELECT
            {rollup_field.name} as name,
            {f'{child_field.name} as child,' if child_field else ''}
            array_remove(array_agg(patent_id), NULL) as patents,
            array_remove(array_agg(regulatory_approval_id), NULL) as regulatory_approvals,
            array_remove(array_agg(trial_id), NULL) as trials
        FROM {entity_category.value} -- intervenable, indicatable or ownable
        WHERE (
            patent_id = ANY($1)
            OR regulatory_approval_id = ANY($2)
            OR trial_id = ANY($3)
        )
        AND {"owner_id IS NOT NULL" if entity_category == EntityCategory.owner else "entity_id IS NOT NULL"}
        AND {rollup_field.name} is not NULL
        AND instance_rollup<>'' -- TODO?
        GROUP BY
            {rollup_field.name},
            {f'{child_field.name}' if child_field else ''}
        """

    async with prisma_context(300) as db:
        results = await db.query_raw(
            query,
            doc_id_map["patents"],
            doc_id_map["regulatory_approvals"],
            doc_id_map["trials"],
        )

    ents_with_docs = [EntWithDocResult(**r) for r in results]

    return ents_with_docs


ENTITY_DOC_LIMIT = 20000


async def get_matching_docs(p: DocumentSearchParams) -> DocsByType:
    """
    Gets docs by type, matching doc_ids
    """

    common_params = {
        "limit": ENTITY_DOC_LIMIT,
        "skip_cache": True,
    }

    regulatory_approvals = asyncio.create_task(
        regulatory_approval_client.search(
            ApprovalParams.parse(p, include={"applicant": True}, **common_params)
        )
    )
    patents = asyncio.create_task(
        patent_client.search(
            PatentParams.parse(p, include={"assignees": True}, **common_params)
        )
    )
    trials = asyncio.create_task(
        trial_client.search(
            TrialParams.parse(p, include={"sponsor": True}, **common_params)
        )
    )

    await asyncio.gather(regulatory_approvals, patents, trials)

    return DocsByType(
        regulatory_approvals={r.id: r for r in regulatory_approvals.result()},
        patents={p.id: p for p in patents.result()},
        trials={t.id: t for t in trials.result()},
    )


def _get_docs_for_ent(
    ewds: list[EntWithDocResult], docs_by_type: DocsByType
) -> dict[str, list]:
    return {
        f"{doc_type}s": compact(
            [
                getattr(docs_by_type, f"{doc_type}s").get(id)
                for id in flatten([getattr(ewd, f"{doc_type}s") for ewd in ewds])
            ]
        )
        for doc_type in DocType.__members__
        if doc_type != DocType.all.name
    }


async def _search(p: EntitySearchParams) -> list[Entity]:
    """
    Internal search for documents grouped by entity
    """
    start = time.monotonic()

    # full docs (if it were pulled in the prior query, would pull dups; thus like this.)
    docs_by_type = await get_matching_docs(DocumentSearchParams.parse(p))

    doc_id_map = {k: list(v.keys()) for k, v in docs_by_type.model_dump().items()}

    # entity/doc matching for ents in first order docs
    ent_with_docs = await get_docs_by_entity_id(
        doc_id_map,
        TermField.category_rollup,
        TermField.instance_rollup,
        p.entity_category,
    )

    grouped_ents = group_by(ent_with_docs, lambda ewd: ewd.name)

    documents_by_entity = [
        Entity.create(
            id=rollup,
            **_get_docs_for_ent(ewds, docs_by_type),
            name=ewds[0].name,
            children=sorted(
                [
                    Entity.create(
                        id=rollup + ewd.child,
                        **_get_docs_for_ent([ewd], docs_by_type),
                        children=[],
                        is_child=True,
                        name=ewd.child,
                        end_year=p.end_year,
                        start_year=p.start_year,
                    )
                    for ewd in ewds
                    if ewd.child and ewd.child != ewd.name
                ],
                key=lambda e: e.record_count,
                reverse=True,
            ),
            end_year=p.end_year,
            is_child=False,
            start_year=p.start_year,
        )
        for rollup, ewds in grouped_ents.items()
    ]

    entities = sorted(documents_by_entity, key=lambda e: e.record_count, reverse=True)[
        0 : p.limit
    ]

    # half the time is in fetching all docs, the other half in transmitting.
    logger.info("Entity search took %s seconds", round(time.monotonic() - start))
    return entities


async def search(p: EntitySearchParams) -> list[Entity]:
    """
    Search for documents, grouped by entity
    """
    search_criteria = EntitySearchParams.parse(p)
    key = get_id(
        {
            **search_criteria.__dict__,
            "api": "entities",
        }
    )
    search_partial = partial(_search, search_criteria)

    # not caching for now; persist to s3 takes too long and it isn't worth.
    if True or p.skip_cache == True:
        entities = await search_partial()
        return entities

    return await retrieve_with_cache_check(
        search_partial,
        key=key,
        decode=lambda str_data: [Entity.load(**p) for p in storage_decoder(str_data)],
    )
