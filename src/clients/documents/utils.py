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
from clients.low_level.prisma import prisma_context
from constants.core import DEFAULT_VECTORIZATION_MODEL
import logging

from typings import TermField
from typings.client import DocumentSearchCriteria, TermSearchCriteria
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


def get_term_clause(
    p: TermSearchCriteria,
    return_type: Type[T],
) -> T:
    """
    Get term search clause
        - look for each term in any of the term fields / mapping tables
        - AND/OR according to query type

    TODO:
    - performance???
    - Use tsvector as soon as https://github.com/RobertCraigie/prisma-client-py/issues/52
    """
    base_mapping_tables = {
        "indications": "some",
        "interventions": "some",
    }

    # choose mapping tables (from which we look for term matches) based on return type
    if return_type == PatentWhereInput or return_type == PatentWhereInputRecursive1:
        mapping_tables = {**base_mapping_tables, "assignees": "some"}
    elif return_type == TrialWhereInput or return_type == TrialWhereInputRecursive1:
        mapping_tables = {**base_mapping_tables, "sponsor": "is"}
    elif return_type == RegulatoryApprovalWhereInput:
        mapping_tables = {**base_mapping_tables, "applicant": "is"}
    else:
        raise ValueError(f"Unsupported return type: {return_type}")

    # get predictate for a given term and term field
    def get_predicates(term: str, term_field: TermField):
        return flatten(
            [
                {table: {comp: {term_field.name: {"equals": term}}}}
                for table, comp in mapping_tables.items()
            ]
        )

    # OR all predicates for a given term, across all term_fields.
    term_clause = [
        {
            "OR": flatten(
                [
                    get_predicates(term.lower(), term_field)
                    for term_field in p.term_fields
                ]
            )
        }
        for term in p.terms
    ]

    # then ANDing or ORing those clauses will abide by the desired query type
    if p.query_type == "AND":
        return cast(T, {"AND": term_clause})

    return cast(T, {"OR": term_clause})


async def get_description_ids(description: str, k: int, doc_type: DocType) -> list[str]:
    """
    Get patent ids within K nearest neighbors of a vectorized description

    Args:
        description (str): a description of the desired patents
        k (int): k nearest neighbors
    """
    # lazy import
    from core.ner.spacy import get_transformer_nlp

    logger.info("Searching documents by description (slow-ish)")
    nlp = get_transformer_nlp(DEFAULT_VECTORIZATION_MODEL)
    vector = nlp(description).vector.tolist()

    query = f"""
        SELECT id FROM {doc_type.name}
        ORDER BY (1 - (vector <=> '{vector}')) DESC
        LIMIT {k}
    """

    async with prisma_context(300) as db:
        results = await db.query_raw(query)
    return [r["id"] for r in results]


def get_search_clause(
    doc_type: DocType,
    p: DocumentSearchCriteria,
    description_ids: Sequence[str] | None,
    return_type: Type[T],
) -> T:
    """
    Get search clause
    """
    date_field = DOC_TYPE_DATE_MAP[doc_type]
    term_clause = get_term_clause(p, return_type)

    where = {
        "AND": [
            term_clause,
            {
                date_field: {
                    "gte": datetime(p.start_year, 1, 1),
                    "lte": datetime(p.end_year, 1, 1),
                }
            },
            {"id": {"in": list(description_ids)}} if description_ids else {},
        ],
    }

    return cast(T, where)
