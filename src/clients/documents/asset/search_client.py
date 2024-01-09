"""
Entity client
"""
from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict
from prisma import Prisma
from pydash import compact, flatten
import logging
from prisma.models import (
    RegulatoryApproval,
    Trial,
)
from pydantic import BaseModel

from clients.low_level.prisma import get_prisma_client
from typings.client import AssetSearchParams
from typings.core import Dataclass
from typings.entities import Entity
from typings.patents import ScoredPatent

from ..patents import find_many as find_patents

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
    id: int | None = None
    name: str = ""


async def get_docs_by_entity_id(
    terms: Sequence[str], client: Prisma
) -> tuple[list[EntWithDocResult], list[str]]:
    """
    Gets documents by entity
    """

    def get_query(table) -> str:
        return f"""
            SELECT
                entity_id, canonical_name,
                patent_id, regulatory_approval_id, trial_id
            FROM {table}
            WHERE canonical_name = ANY($1)
        """

    result = await client.query_raw(
        f"""
        SELECT
            entity_id as id,
            canonical_name,
            array_remove(array_agg(patent_id), NULL) as patents,
            array_remove(array_agg(regulatory_approval_id), NULL) as regulatory_approvals,
            array_remove(array_agg(trial_id), NULL) as trials
        FROM ({' UNION ALL '.join([get_query(table) for table in MAP_TABLES])}) s
        GROUP BY canonical_name, id
        """,
        terms,
    )
    ents_with_docs = [EntWithDocResult(**r) for r in result]
    doc_ids: list[str] = flatten(
        [
            v
            for res in ents_with_docs
            for k, v in res.__dict__.items()
            if k in DocResults.__annotations__.keys()
        ]
    )

    return ents_with_docs, doc_ids


async def get_matching_docs(doc_ids: list[str]) -> DocsByType:
    """
    Gets docs by type, matching doc_ids
    """
    regulatory_approvals = await RegulatoryApproval.prisma().find_many(
        where={"id": {"in": doc_ids}},
        include={"interventions": True, "indications": True},
        take=1000,
    )
    patents = await find_patents(
        where={"id": {"in": doc_ids}},
        include={"interventions": True, "indications": True},
        take=1000,
    )
    trials = await Trial.prisma().find_many(
        where={"id": {"in": doc_ids}},
        include={"interventions": True, "indications": True},
        take=1000,
    )

    # TODO: centralize the prisma client + transform (e.g. to ScoredPatent)
    return DocsByType(
        regulatory_approvals={r.id: r for r in regulatory_approvals},
        patents={p.id: p for p in patents},
        trials={t.id: t for t in trials},
    )


async def _search(terms: Sequence[str]) -> list[Entity]:
    """
    Internal search for documents grouped by entity
    """
    async with get_prisma_client(300) as client:
        ent_with_docs, doc_ids = await get_docs_by_entity_id(terms, client)
        doc_by_type = await get_matching_docs(doc_ids)

    documents_by_entity = [
        Entity(
            id=ent_with_doc.id or 0,
            name=ent_with_doc.name,
            **{
                k: compact([doc_by_type[k][id] for id in ent_with_doc.__dict__[k]])
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
