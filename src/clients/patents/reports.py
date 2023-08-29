"""
Patent client
"""

from typing import Any, Callable, Sequence, cast
import polars as pl
import logging

from typings.patents import PatentApplication

from .types import (
    PatentsReport,
    PatentsReportRecord,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def aggregate(
    patents: Sequence[PatentApplication],
    x_dimensions: list[str],
    y_dimensions: list[str] | None = None,
    x_transform: Callable[[Any], Any] = lambda x: x,
    y_transform: Callable[[Any], Any] = lambda y: y,
) -> Sequence[PatentsReport]:
    """
    Aggregate summary stats

    Args:
        patents (pl.Sequence[PatentApplication]): list of patent applications
        x_dimensions (list[str]): list of x dimensions to aggregate
        y_dimensions ((list[str], optional): y dimensions to aggregate. Defaults to None.

    TODO: SQL for agg?

    Usage:
    ```
    import system; system.initialize();
    from clients import patents as patent_client
    patents = patent_client.search(["asthma"])
    from clients.patents.constants import DOMAINS_OF_INTEREST
    from clients.patents.reports import aggregate
    summaries = aggregate(patents, DOMAINS_OF_INTEREST)
    disease_over_time = aggregate(patents, DOMAINS_OF_INTEREST, ["priority_date"], y_transform=lambda y: y.year)
    ```
    """

    def _aggregate(
        df: pl.DataFrame, x_dim: str, y_dim: str, LIMIT: int = 100
    ) -> list[PatentsReportRecord]:
        if len(y_dim) > 0:
            col_df = (
                # apply y_transform; keep y around
                df.with_columns(pl.col(y_dim).apply(y_transform).alias("y"))
                .select(
                    pl.col(x_dim).apply(x_transform, skip_nulls=False).alias("x"),
                    pl.col("y"),
                )
                .explode("x")
                .drop_nulls()
            )
        else:
            col_df = df.select(
                pl.col(x_dim)
                .explode()
                .drop_nulls()
                .apply(x_transform, skip_nulls=False)
                .alias("x")
            )

        if col_df.shape[0] == 0:
            logger.debug("X %s is empty", x_dim)
            return []

        grouped = (
            col_df.groupby(["x", "y"] if len(y_dim) > 0 else ["x"])
            .agg(pl.count())
            .sort("count")
            .reverse()
            .limit(LIMIT)
        )
        return cast(list[PatentsReportRecord], grouped.to_dicts())

    patent_df = pl.DataFrame(patents, infer_schema_length=None)
    summaries = [
        {"x": x_dim, "y": y_dim, "data": _aggregate(patent_df, x_dim, y_dim)}
        for x_dim in x_dimensions
        for y_dim in (y_dimensions or [""])
    ]

    return cast(list[PatentsReport], summaries)
