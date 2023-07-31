"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl
import logging

from typings import PatentApplication

from .score import calculate_score
from .utils import (
    clean_assignee,
    get_patent_years,
    get_patent_attributes,
)


def format_search_result(
    results: Sequence[dict[str, Any]]
) -> Sequence[PatentApplication]:
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
        pl.col("assignees").apply(
            lambda r: [clean_assignee(assignee) for assignee in r]
        ),
        pl.col("title").map(lambda t: get_patent_attributes(t)).alias("attributes"),
    )

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = calculate_score(df).sort("search_score").reverse()

    return cast(Sequence[PatentApplication], df.to_dicts())
