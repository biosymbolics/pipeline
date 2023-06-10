"""
yfinance client
"""
from datetime import date
import yfinance as yf
import logging


def __normalize(data):
    """
    Normalize data
    - turn date index into column of type str
    """
    data = data.reset_index().rename(columns={"index": "Date"})
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    return data


def fetch_yfinance_data(ticker: str, start_date: date, end_date: date):
    """
    Get yfinance data

    Args:
        ticker (str): ticker to get data for
        start_date (date): start date
        end_date (date): end date
    """
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    logging.info(
        "Fetching yfinance data for %s, %s-%s", ticker, start_date_str, end_date_str
    )
    data = yf.download(ticker, start=start_date_str, end=end_date_str)

    return __normalize(data)
