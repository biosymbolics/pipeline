"""
Entity client
"""
import asyncio
from dataclasses import dataclass
from typing import Literal, Sequence
from prisma import Prisma
from pydash import compact
import logging
from prisma.models import RegulatoryApproval, Trial
from pydantic import BaseModel

from clients.low_level.prisma import prisma_context
from typings.client import AssetSearchParams
from typings.core import Dataclass
from typings.entities import Entity
from typings.documents.patents import ScoredPatent

from ..approvals import find_many as find_regulatory_approvals
from ..patents import find_many as find_patents
from ..trials import find_many as find_trials

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class DocsByType(Dataclass):
    patents: dict[str, ScoredPatent]
    regulatory_approvals: dict[str, RegulatoryApproval]
    trials: dict[str, Trial]


MapTables = Literal["intervenable", "indicatable"]
MAP_TABLES: list[MapTables] = ["intervenable", "indicatable"]


class DocResults(BaseModel):
    patents: list[str] = []
    regulatory_approvals: list[str] = []
    trials: list[str] = []


class EntWithDocResult(DocResults):
    id: int
    name: str


async def get_doc_ids(terms: Sequence[str], client: Prisma) -> list[str]:
    def get_doc_query(table) -> str:
        return f"""
            SELECT COALESCE(patent_id, regulatory_approval_id, trial_id) as id
            FROM {table}
            WHERE canonical_name = ANY($1)
        """

    doc_result = await client.query_raw(
        f"""
        SELECT distinct id
        FROM ({' UNION ALL '.join([get_doc_query(table) for table in MAP_TABLES])}) s
        """,
        terms,
    )

    return [r["id"] for r in doc_result]


async def get_docs_by_entity_id(
    doc_ids: Sequence[str], client: Prisma
) -> list[EntWithDocResult]:
    """
    Gets documents by entity
    """
    result = await client.query_raw(
        f"""
        SELECT
            entity_id as id, canonical_name as name,
            array_remove(array_agg(patent_id), NULL) as patents,
            array_remove(array_agg(regulatory_approval_id), NULL) as regulatory_approvals,
            array_remove(array_agg(trial_id), NULL) as trials
        FROM intervenable
        WHERE COALESCE(patent_id, regulatory_approval_id, trial_id) = ANY($1)
        AND entity_id is not NULL
        GROUP BY canonical_name, entity_id
        """,
        doc_ids,
    )

    ents_with_docs = [EntWithDocResult(**r) for r in result]

    return ents_with_docs


async def get_matching_docs(doc_ids: list[str]) -> DocsByType:
    """
    Gets docs by type, matching doc_ids
    """

    regulatory_approvals = asyncio.create_task(
        find_regulatory_approvals(
            where={"id": {"in": doc_ids}},
            include={"interventions": True, "indications": True},
        )
    )
    patents = asyncio.create_task(
        find_patents(
            where={"id": {"in": doc_ids}},
            include={"assignees": True, "interventions": True, "indications": True},
        )
    )
    trials = asyncio.create_task(
        find_trials(
            where={"id": {"in": doc_ids}},
            include={"interventions": True, "indications": True, "sponsor": True},
        )
    )

    await asyncio.gather(regulatory_approvals, patents, trials)

    # TODO: centralize the prisma client + transform (e.g. to ScoredPatent)
    return DocsByType(
        regulatory_approvals={r.id: r for r in regulatory_approvals.result()},
        patents={p.id: p for p in patents.result()},
        trials={t.id: t for t in trials.result()},
    )


async def _search(terms: Sequence[str]) -> list[Entity]:
    """
    Internal search for documents grouped by entity
    """
    async with prisma_context(300) as client:
        # doc ids that match the suppied terms
        docs_ids = await get_doc_ids(terms, client)

        # full docs (if pulled in the prior query, might pull duplicates)
        doc_by_type = await get_matching_docs(docs_ids)

        # entity/doc matching for ents in first order docs
        ent_with_docs = await get_docs_by_entity_id(docs_ids, client)

    documents_by_entity = [
        Entity(
            id=ent_with_doc.id,
            name=ent_with_doc.name,
            **{
                k: compact(
                    [doc_by_type.__dict__[k].get(id) for id in ent_with_doc.__dict__[k]]
                )
                for k in DocResults.__annotations__.keys()
            },
        )
        for ent_with_doc in ent_with_docs
    ]

    return documents_by_entity


async def search(params: AssetSearchParams) -> list[Entity]:
    """
    Search for documents, grouped by entity
    """

    documents_by_entity = await _search(params.terms)

    logger.info("Got %s assets", len(documents_by_entity))

    return sorted(documents_by_entity, key=lambda e: e.record_count, reverse=True)
