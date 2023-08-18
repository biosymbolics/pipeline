"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl
import logging
from clients.patents.constants import DOMAINS_OF_INTEREST

from clients.patents.types import SearchResults

from typings import PatentApplication

from .score import calculate_score
from .types import PatentsSummary, PatentsSummaryRecord
from .utils import get_patent_years

logger = logging.getLogger(__name__)


def summarize_patents(
    patent_df: pl.DataFrame, columns: list[str]
) -> list[PatentsSummary]:
    """
    Aggregate summary stats

    Args:
        patent_df (pl.DataFrame): dataframe of patent applications
        column_map (dict[str, bool]): map of column names to whether they have hrefs
    """

    def aggregate(
        patent_df: pl.DataFrame, column: str, LIMIT: int = 100
    ) -> list[PatentsSummaryRecord] | None:
        df = patent_df.select(pl.col(column).explode().drop_nulls().alias("term"))
        if df.shape[0] > 0:
            grouped = (
                df.groupby("term").agg(pl.count()).sort("count").reverse().limit(LIMIT)
            )
            return cast(list[PatentsSummaryRecord], grouped.to_dicts())
        else:
            logger.error("Column %s is empty", column)
            return []

    return [
        {"column": column, "data": aggregate(patent_df, column)} for column in columns
    ]


def format_search_result(results: Sequence[dict[str, Any]]) -> SearchResults:
    """
    Format patent search results and adds scores

    Args:
        results (SearchResults): patents search results & summaries
    """

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(results)
    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = df.with_columns(
        *[
            pl.struct(["terms", "domains"])
            .apply(
                lambda rec: [
                    z[0] for z in zip(rec["terms"], rec["domains"]) if z[1] == d  # type: ignore
                ]
            )
            .alias(d)
            for d in DOMAINS_OF_INTEREST
        ]  # type: ignore
    )
    df = calculate_score(df).sort("search_score").reverse()

    summaries = summarize_patents(
        df,
        [
            *DOMAINS_OF_INTEREST,
            "ipc_codes",
            "similar",
        ],
    )

    return {
        "patents": cast(Sequence[PatentApplication], df.to_dicts()),
        "summaries": summaries,
    }
