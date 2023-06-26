"""
Average clinical development pipeline
"""
import streamlit as st
import matplotlib.pyplot as plt
import polars as pl
from langchain.output_parsers import ResponseSchema
import logging

from clients.openai.gpt_client import GptApiClient

st.set_page_config(page_title="Typical ClinDev", page_icon="📈")

st.title("Typical ClinDev Pipeline by Indication")
indication = st.text_input("Enter in a disease or indication", "asthma")

prompt = (
    f"What is the typical clinical development timeline for indication {indication}? "
    "Return the answer as an array of json objects with the following fields: stage, offset, median_duration, iqr. "
)


response_schemas = [
    ResponseSchema(name="stage", description="e.g. Phase 1"),
    ResponseSchema(
        name="offset",
        description="equal to cumulative median duration of previous stages, 0 for the first stage. data type: float.",
    ),
    ResponseSchema(
        name="median_duration",
        description="median duration of this stage in years, e.g. 2.5. data type: float.",
    ),
    ResponseSchema(
        name="iqr",
        description="interquartile range of this stage's duration in years, e.g. 0.8. data type: float.",
    ),
]
df_schema = {
    "stage": pl.Utf8,
    "offset": pl.Float64,
    "median_duration": pl.Float64,
    "iqr": pl.Float64,
}

gpt_client = GptApiClient(schemas=response_schemas)

if st.button("Submit"):
    status = st.progress(0)
    if not indication.strip():
        st.error(f"Please provide an indication.")
    else:
        try:
            with st.spinner("Please wait..."):
                answer_as_array: list[dict] = gpt_client.query(prompt, is_array=True)
                st.code(answer_as_array, "json")
                df = pl.from_dicts(answer_as_array, schema=df_schema).reverse()
                st.dataframe(
                    df.reverse(),
                    column_config={
                        "_index": "",
                        "0": "phase",
                        "1": "offset",
                        "2": "median_duration",
                        "3": "iqr",
                    },
                )
                fig, ax = plt.subplots()
                plt.barh(
                    y=df["stage"], width=df["median_duration"], left=df["offset"] + 1
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"An error occurred: {e}")
