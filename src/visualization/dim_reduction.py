"""
Viz of dimensional reductions
"""
import polars as pl
import streamlit as st
import logging
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.palettes import all_palettes

from .utils import caculate_umap_embedding, get_topics, preprocess_with_tfidf

N_TOPICS = 10  # TODO: coherence model - https://www.kaggle.com/code/yohanb/nmf-visualized-using-umap-and-bokeh/notebook
N_TOP_WORDS = 15


def render_umap(df: pl.DataFrame, n_topics: int = N_TOPICS):
    """
    Render a UMAP plot of a dataframe

    Args:
        df (pl.DataFrame): Dataframe
        n_topics (int, optional): Number of topics. Defaults to N_TOPICS.
    """
    logging.info("Prepping data for UMAP")
    _, tfidf, tfidf_vectorizer, _ = preprocess_with_tfidf(df)

    embedding = caculate_umap_embedding(tfidf)

    topics, nmf_embedding, nmf = get_topics(tfidf, tfidf_vectorizer, n_topics)
    embedding = embedding.with_columns(
        pl.lit(nmf_embedding.argmax(axis=1)).alias("hue")
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
    p.add_tools(hover)  # type: ignore

    st.bokeh_chart(p, use_container_width=True)

    st.subheader("Topics:")
    for topic_idx, topic in enumerate(topics):
        st.write(f"\nTopic {topic_idx}:")
        st.write(topic)
