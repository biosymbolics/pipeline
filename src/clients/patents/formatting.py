"""
Patent client
"""
from typing import Any, Sequence, cast
import polars as pl
import logging
import concurrent.futures

from common.ner.ner import NerTagger, extract_named_entities
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
    tagger = NerTagger()

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

    # Parallelize NER task
    titles = df.select(pl.col("title")).to_series()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        ners = pl.Series(
            executor.map(lambda t: extract_named_entities([str(t)], tagger), titles)
        ).alias("ner")
        df = df.with_columns(ners)

    df = df.with_columns(get_patent_years("priority_date").alias("patent_years"))
    df = calculate_score(df).sort("search_score").reverse()

    return cast(Sequence[PatentApplication], df.to_dicts())
