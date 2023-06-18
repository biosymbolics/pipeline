"""
Summary viz
"""
from typing import Optional, cast
import polars as pl
import streamlit as st
import altair as alt

from common.utils.dataframe import find_string_array_columns

from .colors import Bokeh8

LIMIT = 20
NUM_PER_ROW = 2


def __get_summary_chart(df: pl.DataFrame, column: str, idx: int) -> alt.Chart:
    """
    Get a summary chart for a column
    """
    counts_by_column = (
        df.select(pl.col(column).explode())
        .groupby(column)
        .agg(pl.count())
        .sort("count")
        .reverse()
        .limit(LIMIT)
        .to_pandas()
    )
    chart = (
        alt.Chart(counts_by_column, title=column)
        .mark_bar()
        .encode(
            x=alt.X("count", axis=alt.Axis(title="")),
            y=alt.X(column, axis=alt.Axis(labelLimit=200, title=""), sort="-x"),
            color=alt.value(Bokeh8[idx % len(Bokeh8)]),
        )
    )
    return chart


def render_summary(df: pl.DataFrame, columns_to_summarize: Optional[list[str]] = None):
    """
    Render summary stats

    Args:
        df (pl.DataFrame): Dataframe
        columns_to_summarize (list[str], optional): Columns to summarize. Defaults to all columns containing string arrays.
    """
    columns = (
        find_string_array_columns(df)
        if not columns_to_summarize
        else columns_to_summarize
    )

    charts = [
        __get_summary_chart(df, column, idx) for idx, column in enumerate(columns)
    ]

    # rows of charts
    rows = [(charts[i], charts[i + 1]) for i in range(0, len(charts), NUM_PER_ROW)]
    chart = alt.vconcat(*[alt.hconcat(*row) for row in rows])
    st.altair_chart(cast(alt.Chart, chart))
