"""
Document report client
"""

import asyncio
import logging
import polars as pl


from clients.documents.reports.utils import apply_dimension_limit
from clients.low_level.prisma import prisma_context
from constants.core import SEARCH_TABLE
from typings.client import DocumentSearchParams
from typings.documents.common import DocType

from .constants import X_DIMENSIONS, Y_DIMENSIONS
from .types import Aggregation, CartesianDimension, DocumentReport, DocumentReportRecord

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XYReport:
    @staticmethod
    def _form_query(
        x: str,
        y: str | None,
        # field against which to search, e.g. canonical_name or instance_rollup
        # term_fields: Sequence[TermField],
        aggregation: Aggregation,
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

        # if x is an entity (indicatable, intervenable, ownable), use materialized view (created by ETL)
        if x_info.is_entity:
            entity_join = f"JOIN {SEARCH_TABLE} as entities on entities.{doc_type.name}_id={doc_type.name}.id"
            x_t = f"entities.{x_t}"
            filter = f"WHERE entities.{filter}" if filter else ""
        else:
            entity_join = ""
            filter = f"WHERE {filter}" if filter else ""

        return f"""
            SELECT
                {x_t} as x,
                {aggregation.func}({aggregation.field}) as {aggregation.func}
                {f', {y_t} as y' if y_t else ''}
            FROM {doc_type.name}
                {entity_join}
            JOIN {SEARCH_TABLE} on {SEARCH_TABLE}.{doc_type.name}_id={doc_type.name}.id
                        AND {SEARCH_TABLE}.search @@ plainto_tsquery('english', $1)
            {filter}
            AND {x_t} is not NULL
            GROUP BY {x_t} {f', {y_t}' if y_t else ''}
            ORDER BY count DESC
        """

    @staticmethod
    async def group_by_xy(
        search_params: DocumentSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        x_title: str | None = None,
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
        y_title: str | None = None,
        aggregation: Aggregation = Aggregation(func="count", field="*"),
        doc_type: DocType = DocType.patent,
        filter: str | None = None,
        impute_missing: bool = True,
        limit: int | None = None,
        limit_dimension: CartesianDimension = "x",
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
            aggregation (Aggregation, optional): aggregation. Defaults to Aggregation(func="count", field="*").
                                                 TODO: support other aggregations, and multiple.
            doc_type (DocType, optional): doc type. Defaults to "patent".
            filter (str, optional): additional filter. Defaults to None.
            impute_missing (bool, optional): ensure that for every x, there is a y.
                                             only applies if y_dimension is set. defaults to True.
        """
        if x_dimension not in X_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid x dimension: {x_dimension}")
        if y_dimension and y_dimension not in Y_DIMENSIONS[doc_type]:
            raise ValueError(f"Invalid y dimension: {y_dimension}")

        # TODO - AND/OR
        term_query = " ".join(search_params.terms)

        query = XYReport._form_query(
            x_dimension,
            y_dimension,
            # term_fields=search_params.term_fields,
            filter=filter,
            aggregation=aggregation,
            doc_type=doc_type,
        )
        logger.debug("Running query for xy report: %s", query)

        async with prisma_context(300) as db:
            results = await db.query_raw(query, term_query)

        if len(results) == 0:
            logger.warning("No results for query: %s, params: %s", query, term_query)
            return DocumentReport(
                data=[], x=x_title or x_dimension, y=y_title or y_dimension
            )

        if y_dimension and impute_missing:
            # pivot into x by [y1, y2, y3, ...] format, fill nulls with 0
            # to ensure every x has a value for every y
            df = (
                pl.DataFrame(
                    [DocumentReportRecord(**r) for r in results],
                    # TODO!! just a temp hack for time charts
                    schema_overrides={"y": pl.Int32},
                )
                .pivot(values=aggregation.func, index="x", columns="y")
                .fill_null(0)
                .melt(id_vars=["x"])
            ).rename({"variable": "y", "value": aggregation.func})
            data = [DocumentReportRecord(**r) for r in df.to_dicts()]
        else:
            data = [DocumentReportRecord(**r) for r in results]

        if limit:
            data = apply_dimension_limit(data, limit_dimension, limit)

        return DocumentReport(
            data=data,
            x=x_title or x_dimension,
            y=y_title or y_dimension,
        )

    @staticmethod
    async def group_by_xy_for_filters(
        filters: dict[str, str],
        search_params: DocumentSearchParams,
        x_dimension: str,  # keyof typeof X_DIMENSIONS
        y_dimension: str | None = None,  # keyof typeof Y_DIMENSIONS
        limit: int | None = None,
        limit_dimension: CartesianDimension = "x",
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
                    limit=limit,
                    limit_dimension=limit_dimension,
                )
            )
            for title, filter in filters.items()
        ]
        await asyncio.gather(*reports)
        return [r.result() for r in reports]
