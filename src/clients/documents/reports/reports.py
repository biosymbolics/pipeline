"""
Patent reports
"""

import asyncio
from typing import Any, Callable, Literal, Sequence
import logging


from clients.documents.patents.types import (
    DocumentReport,
    DocumentReportRecord,
)
from clients.low_level.prisma import get_prisma_client
from typings.client import CommonSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DocType = Literal[
    "patent",
    "regulatory_approval",
    "trial",
]


X_DIMENSIONS = {
    "regulatory_approval": {"canonical_name": ""},
    "patent": {"attributes": "", "canonical_name": "", "similar_patents": ""},
    "trial": {"canonical_name": ""},
}
Y_DIMENSIONS: dict[DocType, dict[str, str]] = {
    "regulatory_approval": {"approval_date": ""},
    "patent": {"country_code": "", "priority_date": ""},
    "trial": {
        "comparison_type": "",
        "design": "",
        "end_date": "",
        "hypothesis_type": "",
        "masking": "",
        "start_date": "",
        "status": "",
        "termination_reason": "",
    },
}


class XYReport:
    @staticmethod
    def _get_entity_join(doc_type: str, filter: str) -> str:
        return f"""
            JOIN (
                select {doc_type}_id, canonical_type, canonical_name
                from intervenable
                where {doc_type}_id is not null and {filter}

                UNION ALL

                select {doc_type}_id, canonical_type, canonical_name
                from indicatable
                where {doc_type}_id is not null AND and {filter}

                UNION ALL

                select {doc_type}_id, canonical_type, canonical_name
                from ownable
                where {doc_type}_id is not null AND and {filter}
            )
        """

    @staticmethod
    async def group_by_xy(
        search_params: CommonSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
        x_transform: Callable[[Any], Any] = lambda x: x,
        y_transform: Callable[[Any], Any] = lambda y: y,
        doc_type: DocType = "patent",
        filter: str | None = None,
        report_type: Literal["entities"] = "entities",
    ) -> DocumentReport:
        """
        Group summary stats by x and optionally y dimension

        Args:
            search_params (SearchParams): search params, as you'd send to the search API.
                This becomes the base result set for the report.
            x_dimension (str): x dimension
            y_dimension (str, optional): y dimension. Defaults to None.
            x_transform (Callable[[Any], Any], optional): transform x dimension. Defaults to lambda x: x.
            y_transform (Callable[[Any], Any], optional): transform y dimension. Defaults to lambda y: y.

        Usage:
        ```
        group_by_xy(
            x_dimension="canonical_name",
            y_dimension="priority_date",
            x_transform=lambda x: f"lower({x})",
            y_transform=lambda y: f"DATE_PART('Year', {y})"
        )
        ```
        """
        if x_dimension not in X_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid x dimension: {x_dimension}")
        if y_dimension and y_dimension not in Y_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid y dimension: {y_dimension}")

        if report_type == "entities":
            _join = XYReport._get_entity_join(doc_type, "canonical_name is not null")
            entity_join = (
                f"LEFT {_join} entities on entities.{doc_type}_id={doc_type}.id"
            )
        else:
            entity_join = ""

        def get_query(x: str, y: str | None) -> str:
            search_subquery = f"""
                SELECT id
                FROM {doc_type}
                {XYReport._get_entity_join(doc_type, '{search_params.context_field} is in $1')} context_entities
                    ON {doc_type}.id=context_entities.{doc_type}_id
            """
            return f"""
                select {x} as x, count(*) as count, {f'{y} as y' if y else ''},
                from {doc_type}
                {entity_join}
                WHERE {doc_type}_id in ({search_subquery})
                {f'AND {filter}' if filter else ''}
                GROUP BY {x} {f', {y}' if y else ''}
                ORDER BY count DESC
            """

        async with get_prisma_client(300) as client:
            x = x_transform(x_dimension)
            y = y_transform(y_dimension) if y_dimension else None
            query = get_query(x, y)
            results = await client.query_raw(query, search_params.terms)

        return DocumentReport(
            x=x_dimension,
            y=y_dimension,
            data=[DocumentReportRecord(**r) for r in results],
        )

    @staticmethod
    async def group_by_xy_for_filters(
        filters: Sequence[str],
        search_params: CommonSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
        x_transform: Callable[[Any], Any] = lambda x: x,
        y_transform: Callable[[Any], Any] = lambda y: y,
    ) -> list[DocumentReport]:
        """
        A helper function that returns a report for each filter
        (used for the standard summaries report)
        """
        reports = [
            asyncio.create_task(
                XYReport.group_by_xy(
                    search_params=search_params,
                    x_dimension=x_dimension,
                    y_dimension=y_dimension,
                    x_transform=x_transform,
                    y_transform=y_transform,
                    filter=filter,
                )
            )
            for filter in filters
        ]
        await asyncio.gather(*reports)
        return [r.result() for r in reports]
