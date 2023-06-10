"""
Average clinical development pipeline
"""
import streamlit as st
import matplotlib.pyplot as plt
import polars as pl
from langchain.output_parsers import ResponseSchema
import logging

from clients import GptApiClient
from system import init

init()

st.set_page_config(page_title="ClinDev", page_icon="📈")

st.title("Ask Biosymbolics.ai")
indication = st.text_input("Enter in a disease or indication", "asthma")

prompt = f"Generate a data structure informing of the typical clinical development path in {indication}."

response_schemas = [
    ResponseSchema(name="stage", description="e.g. Phase 1"),
    ResponseSchema(
        name="offset", description="equal to the median duration of previous stage"
    ),
    ResponseSchema(name="median_duration", description="median duration of this stage"),
    ResponseSchema(
        name="iqr", description="interquartile range of this stage's duration"
    ),
]

gpt_client = GptApiClient(schemas=response_schemas)

if st.button("Submit"):
    status = st.progress(0)
    if not indication.strip():
        st.error(f"Please provide an indication.")
    else:
        try:
            answer = gpt_client.query(prompt)
            st.code(answer, "json")
            df = pl.from_records(answer)
            logging.info(df)

            fig, ax = plt.subplots()
            plt.barh(y=df["stage"], width=df["median_duration"], left=df["offset"] + 1)
            st.pyplot()
        except Exception as e:
            st.error(f"An error occurred: {e}")
