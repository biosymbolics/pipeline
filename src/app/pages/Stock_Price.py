"""
Annotations on top of ticker data

TODO: wildly confabulatory. Probably needs Company entity type.
"""
import streamlit as st
from streamlit.elements.time_widgets import DateWidgetReturn
import plost
from datetime import date
import json

from utils.date import format_date
from utils.misc import dict_to_named_tuple
from clients.finance import fetch_yfinance_data
from core import SourceDocIndex

st.set_page_config(page_title="Stock Price & Events", page_icon="📈")


def __get_date(dwr: DateWidgetReturn) -> date:
    """
    Get date from st DateWidgetReturn
    """
    if isinstance(dwr, date):
        return dwr
    raise ValueError(f"Invalid date: {dwr}")


st.title("Stock prices & events")
ticker = st.text_input("Enter a stock symbol", "PFE")
start_date = __get_date(st.date_input("Start date", date(2020, 1, 1)))
end_date = __get_date(st.date_input("End date", date.today()))

prompt = (
    f"""
    For the pharma company represented by the stock symbol {ticker},
    list important events such as regulatory approvals, trial readouts, acquisitions, reorgs, etc.
    that occurred between dates {format_date(start_date)} and {format_date(end_date)}
    as json in the form """
    + '{ "YYYY-MM-DD": "the event" }.'
)

with st.spinner("Initializing..."):
    si = SourceDocIndex(model_name="GPT4")

if st.button("Submit"):
    if not ticker.strip() or not start_date or not end_date:
        st.error(f"Please ticker, start and end date.")
    else:
        try:
            with st.spinner("Please wait..."):
                stock_data = fetch_yfinance_data(ticker, start_date, end_date)  # type: ignore
                source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
                si_answer = si.query(prompt, source)
                st.code(si_answer, "json")
                plost.line_chart(
                    data=stock_data,
                    x="Date",
                    y="Close",
                    x_annot=json.loads(si_answer),
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
