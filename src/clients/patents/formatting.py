"""
Patent client
"""
from typing import Any, Sequence, TypedDict, cast
import polars as pl
import logging

from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.types import SearchResults
from typings import PatentApplication

from .score import calculate_score
from .types import PatentsSummary, PatentsSummaryRecord
from .utils import get_patent_years

logger = logging.getLogger(__name__)

TermDict = TypedDict("TermDict", {"terms": list[str], "domains": list[str]})


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
            logger.debug("Column %s is empty", column)
            return []

    return [
        {"column": column, "data": aggregate(patent_df, column)} for column in columns
    ]


def filter_terms_by_domain(rec: TermDict, domain: str) -> list[str]:
    """
    Filter terms by domain
    """
    terms = [z[0] for z in list(zip(rec["terms"], rec["domains"])) if z[1] == domain]
    return terms


def format_search_result(results: Sequence[dict[str, Any]]) -> SearchResults:
    """
    Format patent search results and adds scores

    Args:
        results (SearchResults): patents search results & summaries
    """

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(results)
    df = df.with_columns(
        get_patent_years("priority_date").alias("patent_years"),
        *[
            df.select(
                pl.struct(["terms", "domains"])
                .apply(lambda rec: filter_terms_by_domain(rec, d))  # type: ignore
                .alias(d)
            ).to_series()
            for d in DOMAINS_OF_INTEREST
        ],
    ).drop("terms", "domains")

    df = calculate_score(df).sort("search_score").reverse()

    summaries = summarize_patents(df, [*DOMAINS_OF_INTEREST, "ipc_codes", "similar"])

    return {
        "patents": cast(Sequence[PatentApplication], df.to_dicts()),
        "summaries": summaries,
    }
