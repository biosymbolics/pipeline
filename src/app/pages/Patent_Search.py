"""
Patent lookup
"""
import json
import streamlit as st
import polars as pl
from typing import cast
from streamlit_timeline import timeline

from common.utils.date import format_date
from clients import patent_client

st.set_page_config(page_title="Patent Search", page_icon="📜", layout="wide")

st.title("Search for patents")
patent_terms = st.multiselect(
    "Enter in terms for patent search",
    [
        "asthma",
        "depression",
        "anxiety",
        "bipolar disorder",
        "bipolar II disorder",
        "b-cell lymphoma",
    ],
    ["asthma"],
)

if st.button("Submit"):
    if not patent_terms:
        st.error(f"Please enter patent terms.")
    else:
        try:
            patents = patent_client.search(patent_terms)
            df = pl.from_dicts(cast(list[dict], patents)).to_pandas()
            st.metric(label="Results", value=len(df))

            st.dataframe(
                df,
                column_config={
                    "priority_date": st.column_config.DateColumn(
                        "priority date",
                        format="YYYY.MM.DD",
                    ),
                    "patent_years_left": st.column_config.NumberColumn(
                        "patent life",
                        help="Number of years left on patent",
                        format="%d years",
                    ),
                    "url": st.column_config.LinkColumn(
                        "Patent",
                        help="Link to the patent",
                        max_chars=25,
                    ),
                },
                hide_index=True,
                height=600,
            )

            timeline_patents = [
                {
                    "start_date": {
                        "year": format_date(patent["priority_date"], "%Y"),
                        "month": format_date(patent["priority_date"], "%m"),
                    },
                    "text": {
                        "headline": patent["title"],
                        "text": patent["abstract"]
                        + "<br /><br />"
                        + f"<a href=\"{patent['url']}\">See on Google Patents.</a>",
                    },
                }
                for patent in patents
            ]
            timeline({"events": timeline_patents}, height=600)
        except Exception as e:
            st.error(f"An error occurred: {e}")
