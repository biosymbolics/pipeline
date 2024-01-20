"""
Patent reports
"""

import asyncio
import logging
from typing import Sequence
import polars as pl

from clients.documents.patents.types import (
    DocumentReport,
    DocumentReportRecord,
)
from clients.low_level.prisma import prisma_client
from typings.client import DocumentSearchParams
from typings.documents.common import DocType, TermField

from .constants import X_DIMENSIONS, Y_DIMENSIONS
from .types import Aggregation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XYReport:
    @staticmethod
    def _get_entity_subquery(x: str, doc_type: DocType, filter: str) -> str:
        """
        Subquery to use if the x dimension is an entity (indicatable, intervenable, ownable)
        """
        return f"""
            (
                select {doc_type.name}_id, canonical_type, {x}
                from intervenable
                where {doc_type.name}_id is not null AND {filter}

                UNION ALL

                select {doc_type.name}_id, canonical_type, {x}
                from indicatable
                where {doc_type.name}_id is not null AND {filter}

                UNION ALL

                select {doc_type.name}_id, 'OTHER' as canonical_type, {x}
                from ownable
                where {doc_type.name}_id is not null AND {filter}
            )
        """

    @staticmethod
    def _get_search_subquery(doc_type: DocType, filter: str) -> str:
        return XYReport._get_entity_subquery("'NX'", doc_type, filter)

    @staticmethod
    def get_query(
        x: str,
        y: str | None,
        term_fields: Sequence[
            TermField
        ],  # field against which to search, e.g. canonical_name or instance_rollup
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

        # if x is an entity (indicatable, intervenable, ownable), we need a subquery to access that info
        if x_info.is_entity:
            entity_sq = XYReport._get_entity_subquery(
                x, doc_type, f"coalesce(length({x}), 0) > 0"
            )
            entity_join = f"LEFT JOIN {entity_sq} entities on entities.{doc_type.name}_id={doc_type.name}.id"
        else:
            entity_join = ""

        # search join to determine result set over which the report is generated
        # TODO: make identical to search
        search_sq = XYReport._get_search_subquery(
            doc_type, " OR ".join([f"{tf.name} = ANY($1)" for tf in term_fields])
        )

        return f"""
            SELECT
                {x_t} as x,
                {aggregation.func}({aggregation.field}) as {aggregation.func}
                {f', {y_t} as y' if y_t else ''}
            FROM {doc_type.name}
                {entity_join}
            WHERE
                {doc_type.name}_id IN (
                    SELECT id
                    FROM {doc_type.name}
                    JOIN {search_sq} context_entities ON {doc_type.name}.id=context_entities.{doc_type.name}_id
                )
                {f'AND {filter}' if filter else ''}
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

        client = await prisma_client(120)
        query = XYReport.get_query(
            x_dimension,
            y_dimension,
            term_fields=search_params.term_fields,
            filter=filter,
            aggregation=aggregation,
            doc_type=doc_type,
        )
        logger.debug("Running query for xy report: %s", query)
        results = await client.query_raw(query, search_params.terms)

        if len(results) == 0:
            logger.warning("No results for query: %s", query)
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
