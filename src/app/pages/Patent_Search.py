"""
Patent lookup
"""
from pydash import compact
import streamlit as st
import polars as pl
from typing import Optional, cast
import logging
import re

from clients.patents import patent_client
from clients.openai.gpt_client import GptApiClient
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


def __get_default_option(options: list[str], params) -> Optional[str]:
    """
    Get the default option from the query params
    """
    search = params.get("search", [None]).pop()

    if not search:
        return None

    default = [opt for opt in options if opt.lower().startswith(search.lower())]
    return default[0] if default else search


@st.cache_data
def get_options():
    options = patent_client.autocomplete_terms("")
    return options


@st.cache_data
def get_data(terms, min_patent_years, relevancy_threshold):
    if not terms:
        st.error(f"Please enter patent terms.")
        return None

    if not is_relevancy_threshold(relevancy_threshold):
        st.error(f"Invalid relevancy threshold: {relevancy_threshold}")
        return None

    st.info(
        f"""
            Searching for patents with terms: {terms},
            min_patent_years: {min_patent_years},
            relevancy_threshold: {relevancy_threshold}
        """
    )
    patents = patent_client.search(terms, False, min_patent_years, relevancy_threshold)
    return patents


@st.cache_data
def get_description(terms: list[str]) -> str:
    logging.info(f"Getting description for terms: {terms}")
    description = gpt_client.describe_terms(terms, ["biomedical research"])
    return description


query_params = st.experimental_get_query_params()
if "patents" not in st.session_state:
    logging.info("Initializing patents")
    st.session_state.patents = None


select_col, metric_col = st.columns([10, 1])


def render_selector():
    with st.sidebar:
        min_patent_years = st.slider("Minimum Patent Years Left", 0, 20, 10)
        relevancy_threshold = st.select_slider(
            "Term Relevance Threshold",
            options=["very low", "low", "medium", "high", "very high"],
            value="high",
        )

    with select_col:
        options = get_options()
        default_option = __get_default_option(options, query_params)
        terms = st.multiselect(
            "Enter in terms for patent search",
            options=compact([*options, default_option]),
            default=default_option,
        )
        terms = __format_terms(terms)

    if st.button("Search"):
        new_patents = get_data(terms, min_patent_years, relevancy_threshold)
        if new_patents is not None:
            df = pl.from_dicts(cast(list[dict], new_patents))
            if new_patents is not None:
                st.session_state.patents = df

    with metric_col:
        st.metric(
            label="Results",
            value=len(st.session_state.patents)
            if st.session_state.patents is not None
            else 0,
        )
    return terms


st.title("Search for patents")


terms = render_selector()

if st.session_state.patents is not None:
    main_tab, landscape_tab, timeline_tab = st.tabs(["Search", "Landscape", "Timeline"])

    with main_tab:
        selection = render_dataframe(st.session_state.patents)

        if selection is not None and len(selection) > 0:
            columns = st.columns(len(selection))
            for idx, selection in enumerate(selection.to_records()):
                with columns[idx]:
                    render_detail(selection)

    with timeline_tab:
        render_timeline(st.session_state.patents)

    with landscape_tab:
        gpt_client = GptApiClient()
        if terms is not None:
            st.subheader(f"About these terms ({str(len(terms))})")
            with st.spinner("Please wait..."):
                st.write(get_description(terms))

        try:
            st.subheader(f"About these patents")
            column_map = {
                "assignees": True,
                "mechanisms": True,
                "ipc_codes": False,
                "inventors": False,
                "similar": True,
                "compounds": True,
                "diseases": True,
            }
            render_summary(st.session_state.patents, column_map)
            # render_umap(patents, terms)
        except Exception as e:
            logging.error(e)
            st.error("Failed to render")
