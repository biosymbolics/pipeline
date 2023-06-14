"""
UI components for patents
"""
import streamlit as st
from streamlit_timeline import timeline
import polars as pl
from annotated_text import annotated_text, annotation, parameters

from common.utils.date import format_date
from .common import get_markdown_link


def render_timeline(patents: pl.DataFrame):
    """
    Render a timeline of patents in streamlit app

    Args:
        patents (pl.DataFrame): patents to render
    """
    if patents is None:
        st.info("No patents to render")
        return

    timeline_patents = [
        {
            "start_date": {
                "year": format_date(patent["priority_date"], "%Y"),
                "month": format_date(patent["priority_date"], "%m"),
            },
            "text": {
                "headline": patent["title"],
                "text": f"""
                    {patent['abstract']}
                    <br /><br />
                    {', '.join(patent['ipc_codes'])}{' '}
                    (score: {round(patent['score'], 2)})
                    <br /><br />
                    <a href="{patent['url']}">See on Google Patents.</a>
                """,
            },
        }
        for patent in patents.to_dicts()[0:100]
        if patent["score"] > 0.1
    ]
    timeline({"events": timeline_patents}, height=600)


def render_detail(patent: dict):
    """
    Render a patent detail in streamlit app

    Args:
        patent (dict): patent to render
    """
    st.header(patent["title"])
    st.markdown(get_markdown_link(patent["url"], patent["publication_number"]))
    st.markdown(
        "**Attributes**: " + " ".join([f"`{attr}`" for attr in patent["attributes"]])
    )
    st.markdown(
        "**Assignees**: " + " ".join([f"`{attr}`" for attr in patent["assignees"]])
    )
    st.write(patent["abstract"])
    st.divider()

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric(label="Patent Years Left", value=patent["patent_years"])
    mcol2.metric(label="Suitability", value=round(patent["score"], 2))
    mcol3.metric(label="Relevancy", value=round(patent["search_rank"], 2))
