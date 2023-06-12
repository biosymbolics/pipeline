"""
Patent lookup
"""
import streamlit as st
import polars as pl
from typing import cast

from clients import patent_client

st.set_page_config(page_title="Patent Search", page_icon="📜", layout="wide")

st.title("Search for patents")
patent_terms = st.multiselect(
    "Enter in terms for patent search",
    ["asthma", "depression", "anxiety", "bipolar", "bipolar II"],
    ["asthma"],
)

if st.button("Submit"):
    if not patent_terms:
        st.error(f"Please enter patent terms.")
    else:
        try:
            patents = patent_client.search(patent_terms)
            df = pl.from_dicts(cast(list[dict], patents)).to_pandas()
            st.dataframe(
                df,
                column_config={
                    "priority_date": st.column_config.DateColumn(
                        "Priority date",
                        format="YYYY.MM.DD",
                    ),
                    "url": st.column_config.LinkColumn(
                        "Patent",
                        help="Link to the patent",
                        max_chars=25,
                    ),
                },
                hide_index=True,
                height=500,
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
