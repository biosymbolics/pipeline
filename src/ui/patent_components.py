"""
UI components for patents
"""
import streamlit as st
from streamlit_timeline import timeline
import polars as pl

from utils.date import format_date
from typings import PatentApplication

from .common import get_horizontal_list, get_markdown_link


URL_BASE = "/Patent_Search?search="


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


def get_google_patent_url(publication_number: str) -> str:
    """
    Get the Google Patent URL for a given publication number

    Args:
        publication_number (str): publication number
    """
    return "https://patents.google.com/patent/" + publication_number.replace("-", "")


def render_detail(patent: PatentApplication):
    """
    Render a patent detail in streamlit app

    Args:
        patent (PatentApplication): patent to render
    """
    st.header(patent["title"])
    st.markdown(get_markdown_link(patent["url"], patent["publication_number"]))
    st.markdown(get_horizontal_list(patent["attributes"], "Attributes"))
    st.markdown(get_horizontal_list(patent["assignees"], "Assignees"))
    st.write(patent["abstract"] + " " + get_markdown_link(patent["url"], "Read more →"))
    st.divider()

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric(label="Patent Years Left", value=patent["patent_years"])
    mcol2.metric(label="Suitability", value=round(patent["score"], 2))
    mcol3.metric(label="Relevancy", value=round(patent["search_rank"], 2))

    st.divider()
    st.markdown(get_horizontal_list(patent["compounds"], "Compounds", URL_BASE))
    st.markdown(get_horizontal_list(patent["diseases"], "Diseases", URL_BASE))
    st.markdown(get_horizontal_list(patent["genes"], "Genes", URL_BASE))
    st.markdown(get_horizontal_list(patent["mechanisms"], "Mechanisms", URL_BASE))
    st.markdown(get_horizontal_list(patent["ipc_codes"], "IPC Codes"))
    st.divider()
    if len(patent["similar"]) > 0:
        st.subheader("Similar Patents")
        st.markdown(
            "\n".join(
                [
                    "- " + get_markdown_link(get_google_patent_url(similar), similar)
                    for similar in patent["similar"]
                ]
            )
        )


def render_dataframe(pl_df: pl.DataFrame):
    """
    Render a dataframe in streamlit app
    (patent-specific; supports selection)

    Args:
        pl_df (pl.DataFrame): dataframe to render
    """
    if pl_df is None:
        st.info("No results")
        return
    df = pl_df.to_pandas()
    df.insert(0, "selected", False)
    edited_df = st.data_editor(
        df,
        column_config={
            "selected": st.column_config.CheckboxColumn(required=True),
            "priority_date": st.column_config.DateColumn(
                "priority date",
                format="YYYY.MM.DD",
            ),
            "patent_years": st.column_config.NumberColumn(
                "patent yrs",
                help="Number of years left on patent",
                format="%d years",
            ),
            "all_scores": st.column_config.BarChartColumn(
                "scores",
                help="Left: suitability; right: term relevancy",
                width="small",
            ),
        },
        column_order=[
            "selected",
            "publication_number",
            "patent_years",
            "all_scores",
            "title",
            "ner",
            "assignees",
            "attributes",
            "priority_date",
            "ipc_codes",
            "search_score",
        ],
        hide_index=True,
        height=450,
    )
    selected_rows = df[edited_df.selected]
    return selected_rows
