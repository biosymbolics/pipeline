"""
Asset client
"""
import asyncio
from dataclasses import dataclass
from typing import Sequence
from prisma import Prisma
from pydash import compact, flatten, group_by
import logging
from pydantic import BaseModel

from clients.low_level.prisma import prisma_client
from typings.client import AssetSearchParams
from typings.core import Dataclass
from typings.documents.common import ENTITY_MAP_TABLES, EntityMapType, TermField
from typings.assets import Asset
from typings import ScoredPatent, ScoredRegulatoryApproval, ScoredTrial

from ..approvals import find_many as find_regulatory_approvals
from ..patents import find_many as find_patents
from ..trials import find_many as find_trials

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DocsByType(Dataclass):
    patents: dict[str, ScoredPatent]
    regulatory_approvals: dict[str, ScoredRegulatoryApproval]
    trials: dict[str, ScoredTrial]


class DocResults(BaseModel):
    patents: list[str] = []
    regulatory_approvals: list[str] = []
    trials: list[str] = []


class EntWithDocResult(DocResults):
    # id: int
    name: str
    child: str | None = None


async def get_doc_ids(client: Prisma, terms: Sequence[str]) -> list[str]:
    def get_doc_query(table) -> str:
        return f"""
            SELECT COALESCE(patent_id, regulatory_approval_id, trial_id) as id
            FROM {table}
            WHERE canonical_name = ANY($1)
        """

    doc_result = await client.query_raw(
        f"""
        SELECT distinct id
        FROM ({' UNION ALL '.join([get_doc_query(table) for table in ENTITY_MAP_TABLES])}) s
        """,
        terms,
    )

    return [r["id"] for r in doc_result]


async def get_docs_by_entity_id(
    client: Prisma,
    doc_ids: Sequence[str],
    rollup_field: TermField,
    child_field: TermField | None,
    entity_map_type: EntityMapType,
) -> list[EntWithDocResult]:
    """
    Gets documents by entity
    """
    query = f"""
        SELECT
            {rollup_field.name} as name,
            {f'{child_field.name} as child,' if child_field else ''}
            array_remove(array_agg(patent_id), NULL) as patents,
            array_remove(array_agg(regulatory_approval_id), NULL) as regulatory_approvals,
            array_remove(array_agg(trial_id), NULL) as trials
        FROM {entity_map_type.value}
        WHERE COALESCE(patent_id, regulatory_approval_id, trial_id) = ANY($1)
        AND entity_id is not NULL
        GROUP BY
            {rollup_field.name},
            {f'{child_field.name}' if child_field else ''}
        """
    result = await client.query_raw(query, doc_ids)

    ents_with_docs = [EntWithDocResult(**r) for r in result]

    return ents_with_docs


async def get_matching_docs(doc_ids: list[str]) -> DocsByType:
    """
    Gets docs by type, matching doc_ids
    """

    regulatory_approvals = asyncio.create_task(
        find_regulatory_approvals(
            where={"id": {"in": doc_ids}},
            include={},
        )
    )
    patents = asyncio.create_task(
        find_patents(
            where={"id": {"in": doc_ids}},
            include={"assignees": True},
        )
    )
    trials = asyncio.create_task(
        find_trials(
            where={"id": {"in": doc_ids}},
            include={"sponsor": True},
        )
    )

    await asyncio.gather(regulatory_approvals, patents, trials)

    return DocsByType(
        regulatory_approvals={r.id: r for r in regulatory_approvals.result()},
        patents={p.id: p for p in patents.result()},
        trials={t.id: t for t in trials.result()},
    )


async def _search(terms: Sequence[str]) -> list[Asset]:
    """
    Internal search for documents grouped by entity
    """
    client = await prisma_client(300)

    # doc ids that match the suppied terms
    docs_ids = await get_doc_ids(client, terms)

    # full docs (if it were pulled in the prior query, would pull dups; thus like this.)
    docs_by_type = await get_matching_docs(docs_ids)

    # entity/doc matching for ents in first order docs
    ent_with_docs = await get_docs_by_entity_id(
        client,
        docs_ids,
        TermField.instance_rollup,
        TermField.canonical_name,
        EntityMapType.intervention,
    )

    grouped_ents = group_by(ent_with_docs, lambda ewd: ewd.name)

    documents_by_entity = [
        Asset(
            id=rollup,
            name=ewds[0].name,
            children=[
                Asset(
                    id=rollup + ewd.child,
                    name=ewd.child,
                    children=[],
                    patents=compact(
                        [docs_by_type.patents.get(id) for id in ewd.patents]
                    ),
                    regulatory_approvals=compact(
                        [
                            docs_by_type.regulatory_approvals.get(id)
                            for id in ewd.regulatory_approvals
                        ]
                    ),
                    trials=compact([docs_by_type.trials.get(id) for id in ewd.trials]),
                )
                for ewd in ewds
                if ewd.child
            ],
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
            trials=compact(
                [
                    docs_by_type.trials.get(id)
                    for id in flatten([ewd.trials for ewd in ewds])
                ]
            ),
        )
        for rollup, ewds in grouped_ents.items()
    ]

    return documents_by_entity[0:1000]


async def search(params: AssetSearchParams) -> list[Asset]:
    """
    Search for documents, grouped by entity
    """

    documents_by_entity = await _search(params.terms)

    logger.info("Got %s assets", len(documents_by_entity))

    return sorted(documents_by_entity, key=lambda e: e.record_count, reverse=True)
