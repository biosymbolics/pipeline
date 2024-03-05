from datetime import datetime
from typing import Sequence, Type, TypeVar, cast
from pydash import flatten
from prisma.types import (
    PatentWhereInput,
    PatentWhereInputRecursive1,
    RegulatoryApprovalWhereInput,
    TrialWhereInput,
    TrialWhereInputRecursive1,
)
from clients.low_level.prisma import prisma_client, prisma_context
from clients.vector.vector_report_client import VectorReportClient
from constants.core import DEFAULT_VECTORIZATION_MODEL, SEARCH_TABLE
import logging

from typings import TermField
from typings.client import (
    DocumentSearchCriteria,
    QueryType,
    TermSearchCriteria,
    VectorSearchParams,
)
from typings.documents.common import DOC_TYPE_DATE_MAP, DocType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar(
    "T",
    bound=RegulatoryApprovalWhereInput
    | TrialWhereInput
    | PatentWhereInput
    | PatentWhereInputRecursive1
    | TrialWhereInputRecursive1,
)


async def get_doc_ids_for_description(
    description: str, doc_types: Sequence[DocType], search_params: VectorSearchParams
) -> list[str]:
    """
    Get patent ids within K nearest neighbors of a vectorized description

    Args:
        description (str): a description of the desired patents
        doc_type (DocType): the type of document to search
        search_params (VectorSearchParams): search parameters
    """
    # lazy import
    from core.ner.spacy import get_transformer_nlp

    nlp = get_transformer_nlp(DEFAULT_VECTORIZATION_MODEL)
    vector = nlp(description).vector.tolist()
    params = search_params.merge({"vector": vector})

    return await VectorReportClient(document_types=doc_types).get_top_doc_ids(params)


async def get_doc_ids_for_terms(
    terms: Sequence[str], query_type: QueryType, doc_types: Sequence[DocType]
) -> list[str]:
    """
    Get document ids that match terms
    """
    query_joiner = " & " if query_type == "AND" else " | "
    term_query = query_joiner.join([" & ".join(t.split(" ")) for t in terms])
    fields = [f"{doc_type.name}_id" for doc_type in doc_types]
    query = f"""
        SELECT COALESCE({", ".join(fields)}) AS id
        FROM {SEARCH_TABLE}
        WHERE search @@ to_tsquery('english', '{term_query}')
        AND ({" OR ".join(fields)} IS NOT NULL)
    """
    async with prisma_context(300) as db:
        results = await db.query_raw(query)

    return [r["id"] for r in results]


def get_search_clause(
    doc_type: DocType,
    p: DocumentSearchCriteria,
    term_matching_ids: Sequence[str] | None,
    description_ids: Sequence[str] | None,
    return_type: Type[T],
) -> T:
    """
    Get search clause
    """
    date_field = DOC_TYPE_DATE_MAP[doc_type]

    where = {
        "AND": [
            {
                p.query_type: [
                    (
                        {"id": {"in": list(term_matching_ids)}}
                        if term_matching_ids is not None
                        else {}
                    ),
                    (
                        {"id": {"in": list(description_ids)}}
                        if description_ids is not None
                        else {}
                    ),
                ]
            },
            {
                date_field: {
                    "gte": datetime(p.start_year, 1, 1),
                    "lte": datetime(p.end_year, 1, 1),
                }
            },
        ],
    }

    return cast(T, where)
