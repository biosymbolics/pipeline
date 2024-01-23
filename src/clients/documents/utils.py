from typing import Type, TypeVar, cast
from pydash import flatten
from prisma.types import (
    PatentWhereInput,
    PatentWhereInputRecursive1,
    RegulatoryApprovalWhereInput,
    TrialWhereInput,
    TrialWhereInputRecursive1,
)

from typings import TermField
from typings.client import TermSearchCriteria

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
        mapping_tables = base_mapping_tables
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