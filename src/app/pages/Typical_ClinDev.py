"""
Average clinical development pipeline
"""
import altair as alt
import streamlit as st
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
        description="equal to cumulative median duration of previous stages, 0 for the first stage.",
        type="float",
    ),
    ResponseSchema(
        name="median_duration",
        description="median duration of this stage in years (e.g. 2.5)",
        type="float",
    ),
    ResponseSchema(
        name="iqr",
        description="interquartile range of this stage's duration in years (e.g. 0.8)",
        type="float",
    ),
]
df_schema = {
    "stage": pl.Utf8,
    "offset": pl.Float64,
    "median_duration": pl.Float64,
    "iqr": pl.Float64,
}


def get_data() -> pl.DataFrame:
    answer_as_array: list[dict] = gpt_client.query(prompt, is_array=True)
    df = pl.from_dicts(answer_as_array, schema=df_schema).reverse()
    df = df.with_columns(
        pl.col("offset").alias("start"),
        pl.struct(["offset", "median_duration"]).apply(lambda rec: rec["offset"] + rec["median_duration"]).alias("end"),  # type: ignore
    )
    return df.sort("offset")


def render_df(df: pl.DataFrame):
    st.dataframe(
        df,
        column_config={
            "0": "phase",
            "1": "offset",
            "2": "median_duration",
            "3": "iqr",
            "4": "start",
            "5": "end",
        },
        hide_index=True,
    )


def render_chart(df: pl.DataFrame):
    pdf = df.to_pandas()
    max_end = pdf["end"].max() + 2
    base = alt.Chart(pdf, width=450, height=300)
    y = alt.Y("stage", sort=alt.SortField(field="x", order="ascending"))
    error_left = base.mark_errorbar(extent="iqr", ticks=True).encode(
        x="start", xError="iqr", y=y
    )
    error_right = base.mark_errorbar(extent="iqr", ticks=True).encode(
        x="end", xError="iqr", y=y
    )
    bar = base.mark_bar().encode(
        x=alt.X("start"),
        x2="end",
        y=y,
    )
    layered = error_left + error_right + bar
    st.altair_chart(layered)  # type: ignore


gpt_client = GptApiClient(schemas=response_schemas, model="gpt-4")

if st.button("Submit"):
    if not indication.strip():
        st.error(f"Please provide an indication.")
    else:
        try:
            with st.spinner("Please wait..."):
                df = get_data()
                render_df(df)
                render_chart(df)
        except Exception as e:
            st.error(f"An error occurred: {e}")
