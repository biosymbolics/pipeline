"""
Viz of dimensional reductions
"""
import polars as pl
import streamlit as st
import logging
from typing import Optional
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import all_palettes

from common.topic import calculate_umap_embedding, get_topics
from .utils import vectorize_data

N_TOPICS = 15  # TODO: coherence model - https://www.kaggle.com/code/yohanb/nmf-visualized-using-umap-and-bokeh/notebook
N_TOP_WORDS = 20


def render_umap(
    df: pl.DataFrame,
    context_terms: Optional[list[str]] = None,
    n_topics: int = N_TOPICS,
):
    """
    Render a UMAP plot of a dataframe

    Args:
        df (pl.DataFrame): Dataframe
        context_terms (Optional[list[str]], optional): Context terms. Defaults to None.
        n_topics (int, optional): Number of topics. Defaults to N_TOPICS.

    TODO: generalize
    """
    logging.info("Prepping data for UMAP")
    vectorized_data, feature_names = vectorize_data(df)

    topics, topic_embedding, dictionary = get_topics(
        vectorized_data,
        feature_names,
        n_topics,
        N_TOP_WORDS,
        context_terms,
    )

    embedding, centroids = calculate_umap_embedding(vectorized_data, dictionary)
    embedding = embedding.with_columns(
        pl.lit(topic_embedding.argmax(axis=1)).alias("hue")
    )
    my_colors = [all_palettes["Category20"][N_TOPICS][i] for i in embedding["hue"]]

    logging.info("Rendering UMAP")
    source = ColumnDataSource(
        data=dict(
            x=embedding.select(pl.col("x")).to_series().to_list(),
            y=embedding.select(pl.col("y")).to_series().to_list(),
            publication_number=df.select(pl.col("publication_number"))
            .to_series()
            .to_list(),
            title=df.select(pl.col("title")).to_series().to_list(),
            colors=my_colors,
            topic=[topics[i] for i in embedding["hue"]],
            alpha=[0.7] * embedding.shape[0],
            size=[7] * embedding.shape[0],
        ),
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
    )  # type: ignore

    p = figure(plot_width=600, plot_height=600, title="Patents")
    p.circle(
        "x",
        "y",
        size="size",
        fill_color="colors",
        alpha="alpha",
        line_alpha=0,
        line_width=0.01,
        source=source,
        name="df",
        legend_field="topic",
    )
    for i in range(n_topics):
        p.cross(
            x=centroids[i, 0],
            y=centroids[i, 1],
            size=15,
            color="grey",
            line_width=1,
            angle=0.79,
        )
    p.add_tools(hover)  # type: ignore
    p.legend.location = "bottom_left"

    st.bokeh_chart(p, use_container_width=True)
