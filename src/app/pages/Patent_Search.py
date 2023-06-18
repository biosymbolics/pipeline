"""
Patent lookup
"""
import streamlit as st
import polars as pl
from typing import cast
import logging
import re

from clients import patent_client
from clients import GptApiClient

# from visualization.dim_reduction import render_umap
from visualization.summary import render_summary
from ui.patent_components import render_dataframe, render_detail, render_timeline

st.set_page_config(page_title="Patent Search", page_icon="📜", layout="wide")

# increase the max width of chips
st.markdown(
    """
    <style>
        .stMultiSelect [data-baseweb=select] span{
            max-width: 500px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def __format_terms(terms: list[str]) -> list[str]:
    """
    Removes trailing counts
    Example: "asthma (100)" -> "asthma"
    """
    return [re.sub("\\([0-9]{1,}\\)$", "", term).strip() for term in terms]


@st.cache_resource
def get_options():
    return patent_client.autocomplete_terms("")


@st.cache_resource(experimental_allow_widgets=True)
def get_data(options):
    if not options:
        return None
    terms = st.multiselect("Enter in terms for patent search", options=options)
    if not terms:
        st.error(f"Please enter patent terms.")
        return None
    terms = __format_terms(terms)
    df = patent_client.search(terms)
    return pl.from_dicts(cast(list[dict], df)), terms


def render_selector():
    col1, col2 = st.columns([10, 1])
    with col1:
        options = get_options()
        patents, terms = get_data(options) or (None, None)
    with col2:
        st.metric(label="Results", value=len(patents) if patents is not None else 0)

    return patents, terms


st.title("Search for patents")

patents, terms = render_selector()
main_tab, overview_tab, timeline_tab = st.tabs(["Main", "Overview", "Timeline"])

if patents is not None:
    with main_tab:
        selection = render_dataframe(patents)

        if selection is not None and len(selection) > 0:
            columns = st.columns(len(selection))
            for idx, selection in enumerate(selection.to_records()):
                with columns[idx]:
                    render_detail(selection)

    with timeline_tab:
        render_timeline(patents)

    with overview_tab:
        gpt_client = GptApiClient()
        st.write(
            """
            Ideas:
            - What are the commonalities in these patents? (e.g. commonly co-occurring terms)
            - What are the changing themes over time?
            - What are the most common assignees, inventors, diseases, compounds, etc?
            """
        )
        # if terms is not None:
        #     st.subheader(f"About these terms ({str(len(terms))})")
        #     st.write(gpt_client.describe_terms(terms, ["biomedical research"]))

        try:
            if patents is not None:
                st.subheader(f"About these patents ({str(len(patents))})")
                render_summary(patents)
                # render_umap(patents, terms)
        except Exception as e:
            logging.error(e)
            st.error("Failed to render UMAP")
