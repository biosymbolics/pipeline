"""
Summary viz
"""
import logging
from typing import Optional, cast
import polars as pl
import streamlit as st
import altair as alt

from common.utils.dataframe import find_string_array_columns
from common.utils.list import batch

from .colors import Bokeh8

LIMIT = 20


def __get_summary_chart(
    df: pl.DataFrame, column: str, idx: int, has_href=True
) -> alt.Chart:
    """
    Get a summary chart for a column

    Args:
        df (pl.DataFrame): dataframe
        column (str): column name
        idx (int): index of column
    """
    df = (
        df.select(pl.col(column).explode().drop_nulls())
        .groupby(column)
        .agg(pl.count())
        .sort("count")
        .reverse()
        .limit(LIMIT)
    )

    if has_href:
        by_column = df.with_columns(
            ("/Patent_Search?search=" + pl.col(column)).alias("url")
        ).to_pandas()
    else:
        by_column = df.to_pandas()

    chart = (
        alt.Chart(by_column, title=column, width=150)
        .mark_bar()
        .encode(
            x=alt.X("count", axis=alt.Axis(title="")),
            y=alt.Y(column, axis=alt.Axis(labelLimit=300, title=""), sort="-x"),
            color=alt.value(Bokeh8[idx % len(Bokeh8)]),
            tooltip=column,
            href="url",
        )
    )

    chart = chart.interactive()

    return chart


def render_summary(
    df: pl.DataFrame, columns: Optional[list[str]] = None, suppressions: list[str] = []
):
    """
    Render summary stats

    Args:
        df (pl.DataFrame): Dataframe
        columns (list[str], optional): Columns to summarize. Defaults to all columns containing string arrays.
        suppressions (list[str], optional): Columns to suppress. Defaults to [].
    """
    columns = (
        [
            col
            for col in find_string_array_columns(df, allow_empty=False)
            if col not in suppressions
        ]
        if not columns
        else columns
    )

    charts = [
        __get_summary_chart(df, column, idx) for idx, column in enumerate(columns)
    ]

    # rows of charts
    rows = batch(charts, 2)
    chart = alt.vconcat(*[alt.hconcat(*row) for row in rows])
    st.altair_chart(cast(alt.Chart, chart))
