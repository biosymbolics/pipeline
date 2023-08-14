"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl
import logging

from pydash import compact
from clients.patents.types import SearchResults

from typings import PatentApplication

from .score import calculate_score
from .utils import get_patent_years

logger = logging.getLogger(__name__)


def summarize_patents(
    patent_df: pl.DataFrame, columns: list[str]
) -> list[list[dict[str, Any]]]:
    """
    Aggregate summary stats

    Args:
        patent_df (pl.DataFrame): dataframe of patent applications
        column_map (dict[str, bool]): map of column names to whether they have hrefs
    """

    def aggregate_terms(
        patent_df: pl.DataFrame, column: str, LIMIT: int = 100
    ) -> list[dict[str, Any]] | None:
        df = patent_df.select(pl.col(column).explode().drop_nulls())
        if df.shape[0] > 0:
            grouped = (
                df.groupby(column).agg(pl.count()).sort("count").reverse().limit(LIMIT)
            )
            return grouped.to_dicts()
        else:
            logger.error("Column %s is empty", column)
            return None

    return compact([aggregate_terms(patent_df, column) for column in columns])


def format_search_result(results: Sequence[dict[str, Any]]) -> SearchResults:
    """
    Format BigQuery patent search results and adds scores

    Args:
        results (list[dict]): list of search results
    """

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(results)

    df = df.with_columns(
        pl.col("priority_date")
        .cast(str)
        .str.strptime(pl.Date, "%Y%m%d")
        .alias("priority_date"),
    )

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = calculate_score(df).sort("search_score").reverse()

    summaries = summarize_patents(
        df,
        ["assignees", "compounds", "diseaes", "inventors", "ipc_codes", "mechanisms"],
    )

    return {
        "data": cast(Sequence[PatentApplication], df.to_dicts()),
        "summaries": summaries,
    }
