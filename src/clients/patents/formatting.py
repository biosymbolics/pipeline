"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl

from .score import calculate_score
from .utils import (
    clean_assignee,
    get_patent_years,
    get_patent_attributes,
)
from typings import PatentBasicInfo


def format_search_result(
    results: Sequence[dict[str, Any]]
) -> Sequence[PatentBasicInfo]:
    """
    Format BigQuery patent search results and adds scores

    Args:
        results (list[dict]): list of search results
    """
    df = pl.from_dicts(results)

    df = df.with_columns(
        pl.col("priority_date")
        .cast(str)
        .str.strptime(pl.Date, "%Y%m%d")
        .alias("priority_date"),
        pl.col("assignees").apply(
            lambda r: [clean_assignee(assignee) for assignee in r]
        ),
        pl.col("title").map(lambda t: get_patent_attributes(t)).alias("attributes"),
    )

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = calculate_score(df).sort("search_score").reverse()

    return cast(Sequence[PatentBasicInfo], df.to_dicts())
