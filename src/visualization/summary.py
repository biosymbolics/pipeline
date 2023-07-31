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

    chart_encodings = {
        "x": alt.X("count", axis=alt.Axis(title="")),
        "y": alt.Y(column, axis=alt.Axis(labelLimit=300, title=""), sort="-x"),
        "color": alt.value(Bokeh8[idx % len(Bokeh8)]),
        "tooltip": column,
        "href": "url" if has_href else alt.value(None),
    }
    chart = (
        alt.Chart(by_column, title=column, width=150)
        .mark_bar()
        .encode(**chart_encodings)
    )

    chart = chart.interactive()

    return chart


def render_summary(
    df: pl.DataFrame,
    column_map: Optional[dict[str, bool]] = None,
    suppressions: list[str] = [],
):
    """
    Render summary stats

    Args:
        df (pl.DataFrame): Dataframe
        column_map (dict[str, bool], optional): Columns to summarize; value is whether to add href.
            Defaults to None, in which case all string array columns are used.
        suppressions (list[str], optional): Columns to suppress. Defaults to [].
    """
    column_map = column_map or {
        col: True for col in find_string_array_columns(df) if col not in suppressions
    }

    charts = [
        __get_summary_chart(df, column, idx, has_href)
        for idx, (column, has_href) in enumerate(column_map.items())
    ]

    # rows of charts
    rows = batch(charts, 2)
    chart = alt.vconcat(*[alt.hconcat(*row) for row in rows])
    st.altair_chart(cast(alt.Chart, chart))
