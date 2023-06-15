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
from ui.patents import render_detail, render_timeline

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
    return [re.sub("\([0-9]{1,}\)$", "", term).strip() for term in terms]


def dataframe_with_selections(pl_df: pl.DataFrame):
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


@st.cache_resource
def get_options():
    return patent_client.autocomplete_terms("")


@st.cache_resource(experimental_allow_widgets=True)
def get_data(options):
    terms = st.multiselect("Enter in terms for patent search", options=options)
    if not terms:
        st.error(f"Please enter patent terms.")
        return
    terms = __format_terms(terms)
    df = patent_client.search(terms)
    return pl.from_dicts(cast(list[dict], df)), terms


def render_selector():
    col1, col2 = st.columns([10, 1])
    with col1:
        options = get_options()
        patents, terms = get_data(options)
    with col2:
        st.metric(label="Results", value=len(patents) if patents is not None else 0)

    return patents, terms


st.title("Search for patents")

try:
    patents, terms = render_selector()
    main_tab, overview_tab, timeline_tab = st.tabs(["Main", "Overview", "Timeline"])

    with main_tab:
        selection = dataframe_with_selections(patents)

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
            - Summarize the search term(s) with GPT
            """
        )
        if terms is not None:
            st.subheader(f"About these terms ({str(len(terms))})")
            st.write(gpt_client.describe_terms(terms))
except Exception as e:
    st.error(f"An error occurred: {e}")
