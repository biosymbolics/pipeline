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
from clients.patents.types import is_relevancy_threshold

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
def get_data(_terms, _min_patent_years, _relevancy_threshold):
    if st.button("Search"):
        if not _terms:
            st.error(f"Please enter patent terms.")
            return pl.DataFrame()

        if not is_relevancy_threshold(_relevancy_threshold):
            st.error(f"Invalid relevancy decay: {_relevancy_threshold}")
            return pl.DataFrame()

        st.info(
            f"Searching for patents with terms: {_terms}, min_patent_years: {_min_patent_years}, relevancy_threshold: {relevancy_threshold}"
        )
        df = patent_client.search(_terms, _min_patent_years, _relevancy_threshold)
        return pl.from_dicts(cast(list[dict], df))


def render_selector():
    col1, col2 = st.columns([10, 1])
    with col1:
        patents = get_data(terms, min_patent_years, relevancy_threshold)
    with col2:
        st.metric(label="Results", value=len(patents) if patents is not None else 0)

    return patents


@st.cache_resource
def get_description(_terms: list[str]) -> str:
    description = gpt_client.describe_terms(_terms, ["biomedical research"])
    return description


st.title("Search for patents")
with st.sidebar:
    min_patent_years = st.slider("Minimum Patent Years Left", 0, 20, 10)
    relevancy_threshold = st.select_slider(
        "Term Relevance Threshold",
        options=["very low", "low", "medium", "high", "very high"],
        value="high",
    )
options = get_options()
terms = st.multiselect("Enter in terms for patent search", options=options)
terms = __format_terms(terms)

patents = render_selector()
main_tab, landscape_tab, timeline_tab = st.tabs(["Search", "Landscape", "Timeline"])

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

    with landscape_tab:
        gpt_client = GptApiClient()
        if terms is not None:
            st.subheader(f"About these terms ({str(len(terms))})")
            st.write(get_description(terms))

        try:
            st.subheader(f"About these patents ({str(len(patents))})")
            render_summary(patents, None, ["top_terms"])
            # render_umap(patents, terms)
        except Exception as e:
            logging.error(e)
            st.error("Failed to render")
