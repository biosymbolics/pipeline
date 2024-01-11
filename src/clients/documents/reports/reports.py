"""
Patent reports
"""

import asyncio
from dataclasses import dataclass
from typing import Callable, Literal, Sequence
import logging


from clients.documents.patents.types import (
    DocumentReport,
    DocumentReportRecord,
)
from clients.low_level.prisma import prisma_client
from typings.client import CommonSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DocType = Literal[
    "patent",
    "regulatory_approval",
    "trial",
]


@dataclass(frozen=True)
class DimensionInfo:
    is_entity: bool = False
    transform: Callable[[str], str] = lambda x: x


X_DIMENSIONS: dict[DocType, dict[str, DimensionInfo]] = {
    "regulatory_approval": {
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
    },
    "patent": {
        "attributes": DimensionInfo(),
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
        "similar_patents": DimensionInfo(),
    },
    "trial": {
        "name": DimensionInfo(is_entity=True),
        "canonical_name": DimensionInfo(is_entity=True),
        "instance_rollup": DimensionInfo(is_entity=True),
    },
}

Y_DIMENSIONS: dict[DocType, dict[str, DimensionInfo]] = {
    "regulatory_approval": {
        "approval_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
    },
    "patent": {
        "country_code": DimensionInfo(),
        "priority_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
    },
    "trial": {
        "comparison_type": DimensionInfo(),
        "design": DimensionInfo(),
        "end_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
        "hypothesis_type": DimensionInfo(),
        "masking": DimensionInfo(),
        "start_date": DimensionInfo(transform=lambda y: f"DATE_PART('Year', {y})"),
        "status": DimensionInfo(),
        "termination_reason": DimensionInfo(),
    },
}


class XYReport:
    @staticmethod
    def _get_entity_subquery(x: str, doc_type: str, filter: str) -> str:
        """
        Subquery to use if the x dimension is an entity (indicatable, intervenable, ownable)
        """
        return f"""
            (
                select {doc_type}_id, canonical_type, {x}
                from intervenable
                where {doc_type}_id is not null AND {filter}

                UNION ALL

                select {doc_type}_id, canonical_type, {x}
                from indicatable
                where {doc_type}_id is not null AND {filter}

                UNION ALL

                select {doc_type}_id, 'OTHER' as canonical_type, {x}
                from ownable
                where {doc_type}_id is not null AND {filter}
            )
        """

    @staticmethod
    def get_query(
        x: str,
        y: str | None,
        search_field: str,  # field against which to search, e.g. canonical_name or instance_rollup
        doc_type: DocType,
        filter: str | None = None,
    ) -> str:
        """
        Get the query for the report
        """
        x_info = X_DIMENSIONS[doc_type][x]
        y_info = Y_DIMENSIONS[doc_type][y] if y else None
        x_t = x_info.transform(x)
        y_t = y_info.transform(y or "") if y_info else None

        # if x is an entity (indicatable, intervenable, ownable), we need a subquery to access that info
        if x_info.is_entity:
            sq = XYReport._get_entity_subquery(x, doc_type, f"{x} is not null")
            entity_join = (
                f"LEFT JOIN {sq} entities on entities.{doc_type}_id={doc_type}.id"
            )
        else:
            entity_join = ""

        # search join to determine result set for report
        search_sq = XYReport._get_entity_subquery(
            search_field, doc_type, f"{search_field} = ANY($1)"
        )
        search_subquery = f"""
            SELECT id
            FROM {doc_type}
            JOIN {search_sq} context_entities ON {doc_type}.id=context_entities.{doc_type}_id
        """

        return f"""
            select {x_t} as x, count(*) as count {f', {y_t} as y' if y_t else ''}
            from {doc_type}
            {entity_join}
            WHERE {doc_type}_id in ({search_subquery})
            {f'AND {filter}' if filter else ''}
            GROUP BY {x_t} {f', {y_t}' if y_t else ''}
            ORDER BY count DESC
        """

    @staticmethod
    async def group_by_xy(
        search_params: CommonSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        x_title: str | None = None,
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
        y_title: str | None = None,
        doc_type: DocType = "patent",
        filter: str | None = None,
    ) -> DocumentReport:
        """
        Group summary stats by x and optionally y dimension

        Args:
            search_params (SearchParams): search params, as you'd send to the search API.
                This becomes the base result set for the report.
            x_dimension (str): x dimension
            x_title (str, optional): x title. Defaults to None.
            y_dimension (str, optional): y dimension. Defaults to None.
            y_title (str, optional): y title. Defaults to None.
            doc_type (DocType, optional): doc type. Defaults to "patent".
            filter (str, optional): additional filter. Defaults to None.

        Usage:
        ```
        group_by_xy(
            search_params=SearchParams(terms=["asthma"]),
            x_dimension="canonical_name",
            x_title="disease",
            y_dimension="priority_date",
        )
        ```
        """
        if x_dimension not in X_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid x dimension: {x_dimension}")
        if y_dimension and y_dimension not in Y_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid y dimension: {y_dimension}")

        client = await prisma_client(300)
        results = await client.query_raw(
            XYReport.get_query(
                x_dimension,
                y_dimension,
                search_field=search_params.term_field,
                filter=filter,
                doc_type=doc_type,
            ),
            search_params.terms,
        )

        return DocumentReport(
            x=x_title or x_dimension,
            y=y_title or y_dimension,
            data=[DocumentReportRecord(**r) for r in results],
        )

    @staticmethod
    async def group_by_xy_for_filters(
        filters: dict[str, str],
        search_params: CommonSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
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
                    x_title=title,
                    y_dimension=y_dimension,
                    filter=filter,
                )
            )
            for title, filter in filters.items()
        ]
        await asyncio.gather(*reports)
        return [r.result() for r in reports]
