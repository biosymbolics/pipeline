"""
Viz of dimensional reductions
"""
import umap
import polars as pl
import streamlit as st
import logging
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource

from .utils import prep_data_for_umap


def render_umap(df: pl.DataFrame):
    """
    Render a UMAP plot of a dataframe

    Args:
        df (pl.DataFrame): Dataframe
    """
    logging.info("prepping data for UMAP")
    prepped_df = prep_data_for_umap(df)

    combined_df = prepped_df.select(pl.concat_list(pl.col("*")).alias("combined"))
    combined = combined_df.to_series().to_list()

    logging.info("Attempting UMAP")
    umap_embedding = umap.UMAP(
        n_neighbors=5, min_dist=0.3, random_state=42
    ).fit_transform(combined)

    logging.info("Rendering UMAP")

    source = ColumnDataSource(
        data=dict(
            x=umap_embedding[:, 0],
            y=umap_embedding[:, 1],
            publication_number=df.select(pl.col("publication_number"))
            .to_series()
            .to_list(),
            title=df.select(pl.col("title")).to_series().to_list(),
        )
    )

    hover = HoverTool(
        names=["df"],
        tooltips="""
        <div style="margin: 10">
            <div style="margin: 0 auto; width:300px;">
                <span style="font-size: 12px; font-weight: bold;">@publication_number</span>
                <span style="font-size: 12px">@title</span>
            </div>
        </div>
        """,
    )

    p = figure(plot_width=600, plot_height=600, title="Patents")
    p.circle(
        "x",
        "y",
        size=5,
        fill_color="green",
        alpha=0.7,
        line_alpha=0,
        line_width=0.01,
        source=source,
        name="df",
    )
    p.add_tools(hover)

    st.bokeh_chart(p, use_container_width=True)
