"""
Patent client
"""

from typing import Sequence, cast
import polars as pl
import logging


from .types import (
    PatentApplication,
    PatentsSummary,
    PatentsSummaryRecord,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_summaries(
    patents: Sequence[PatentApplication], columns: list[str]
) -> Sequence[PatentsSummary]:
    """
    Aggregate summary stats

    Args:
        patents (pl.Sequence[PatentApplication]): list of patent applications
        column_map (dict[str, bool]): map of column names to whether they have hrefs

    Usage:
    ```
    summaries = generate_summaries(patents, [*DOMAINS_OF_INTEREST, "ipc_codes", "similar"])
    ```
    """

    def aggregate(
        df: pl.DataFrame, column: str, LIMIT: int = 100
    ) -> list[PatentsSummaryRecord]:
        col_df = df.select(pl.col(column).explode().drop_nulls().alias("term"))
        if col_df.shape[0] > 0:
            grouped = (
                col_df.groupby("term")
                .agg(pl.count())
                .sort("count")
                .reverse()
                .limit(LIMIT)
            )
            return cast(list[PatentsSummaryRecord], grouped.to_dicts())
        else:
            logger.debug("Column %s is empty", column)
            return []

    patent_df = pl.DataFrame(patents)
    summaries = [
        {"column": column, "data": aggregate(patent_df, column)} for column in columns
    ]

    return cast(list[PatentsSummary], summaries)
