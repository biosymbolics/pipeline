"""
Patent lookup
"""
import streamlit as st
import polars as pl
from typing import cast
import logging
import re

from clients import patent_client
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
            "patent_years_left": st.column_config.NumberColumn(
                "patent life",
                help="Number of years left on patent",
                format="%d years",
            ),
            "all_scores": st.column_config.BarChartColumn(
                "Scores",
                help="Left: suitability; right: term relevancy",
                width="small",
            ),
            "url": st.column_config.LinkColumn(
                "Patent",
                help="Link to the patent",
                width="medium",
            ),
        },
        column_order=[
            "selected",
            "publication_number",
            "patent_years",
            "all_scores",
            "title",
            "url",
            "assignees",
            "abstract",
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
    df = patent_client.search(__format_terms(terms))
    return pl.from_dicts(cast(list[dict], df))


def render_selector():
    col1, col2 = st.columns([10, 1])
    with col1:
        options = get_options()
        patents = get_data(options)
    with col2:
        st.metric(label="Results", value=len(patents) if patents is not None else 0)

    return patents


st.title("Search for patents")

try:
    patents = render_selector()
    main_tab, timeline_tab = st.tabs(["Main", "Timeline"])

    with main_tab:
        selection = dataframe_with_selections(patents)

        if selection is not None:
            render_detail(selection.to_records()[0])

    with timeline_tab:
        render_timeline(patents)
except Exception as e:
    st.error(f"An error occurred: {e}")
