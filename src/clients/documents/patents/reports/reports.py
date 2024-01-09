"""
Patent reports
"""

from typing import Any, Callable, Sequence, cast
import polars as pl
import logging


from clients.documents.patents.types import (
    PatentsReport,
    PatentsReportRecord,
)
from typings import ScoredPatent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def group_by_xy(
    patents: Sequence[ScoredPatent],
    x_dimensions: list[str],
    y_dimensions: list[str] | None = None,
    x_transform: Callable[[Any], Any] = lambda x: x,
    y_transform: Callable[[Any], Any] = lambda y: y,
) -> list[PatentsReport]:
    """
    Group summary stats by x and optionally y dimensions
    Returns one report per (x_dimension x y_dimension)

    Args:
        patents (pl.Sequence[PatentApplication]): list of patent applications
        x_dimensions (list[str]): list of x dimensions to aggregate
        y_dimensions ((list[str], optional): y dimensions to aggregate. Defaults to None.
        x_transform (Callable[[Any], Any], optional): transform x dimension. Defaults to lambda x: x.
        y_transform (Callable[[Any], Any], optional): transform y dimension. Defaults to lambda y: y.
        rollup_level (RollupLevel, optional): rollup level. Defaults to None.

    TODO: use SQL for agg instead of polars/df?

    Usage:
    ```
    import system; system.initialize();
    from clients import patents as patent_client
    patents = patent_client.search(["asthma"], skip_cache=True)
    from clients.patents.constants import DOMAINS_OF_INTEREST
    from clients.patents.reports import aggregate
    summaries = aggregate(patents, DOMAINS_OF_INTEREST)
    disease_over_time = aggregate(patents, DOMAINS_OF_INTEREST, ["priority_date"], y_transform=lambda y: y.year)
    ```
    """

    def _aggregate(
        df: pl.DataFrame, x_dim: str, y_dim: str | None, LIMIT: int = 100
    ) -> list[PatentsReportRecord]:
        if y_dim is not None:
            # explode y_dim if list
            if df.select(pl.col(y_dim)).dtypes == pl.List:
                df = df.explode(y_dim)

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
            col_df.groupby(["x", "y"] if y_dim is not None else ["x"])
            .agg(pl.count())
            .sort("count")
            .reverse()
            .limit(LIMIT)
        )
        return [PatentsReportRecord(**prr) for prr in grouped.to_dicts()]

    patent_df = pl.from_dicts(
        [p.__dict__ for p in patents],
        schema=[*x_dimensions, *(y_dimensions or [])],
        infer_schema_length=None,
    )
    summaries = [
        PatentsReport(x=x_dim, y=y_dim, data=_aggregate(patent_df, x_dim, y_dim))
        for x_dim in x_dimensions
        for y_dim in (y_dimensions or [None])  # type: ignore
    ]

    return summaries
