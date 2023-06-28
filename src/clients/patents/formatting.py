"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl
import logging
import concurrent.futures

from common.ner.ner import NerTagger
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

    titles = df.select(pl.col("title")).to_series().to_list()

    ners = NerTagger.get_instance().extract(titles, flatten_results=False)
    df = df.with_columns(pl.Series(ners).alias("ner"))

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = calculate_score(df).sort("search_score").reverse()

    return cast(Sequence[PatentApplication], df.to_dicts())
