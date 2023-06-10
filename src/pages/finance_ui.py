import streamlit as st
import plost
from datetime import date
import json

from common.utils.misc import dict_to_named_tuple
from clients import fetch_yfinance_data
from core import SourceDocIndex
from system import init

init()

st.title("Ask Biosymbolic.ai")
ticker = st.text_input("Enter a stock symbol", "PFE")
start_date = st.date_input("Start date", date(2020, 1, 1))
end_date = st.date_input("End date", date.today())

prompt = (
    "Please provide important events that have occurred for the company "
    f"represented by the ticker symbol {ticker} "
    'as json in the form { "YYYY-MM-DD": "what happened" } '
)

if st.button("Submit"):
    status = st.progress(0)
    if not prompt.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            stock_data = fetch_yfinance_data(ticker, start_date, end_date)  # type: ignore
            source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
            si = SourceDocIndex()
            si_answer = si.query(prompt, source)
            st.code(si_answer, "json")
            plost.line_chart(
                data=stock_data,
                x="date:Date",
                y="Close",
                x_annot=json.loads(si_answer),
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
