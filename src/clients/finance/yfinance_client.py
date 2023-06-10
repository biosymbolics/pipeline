"""
yfinance client
"""
from datetime import date
import yfinance as yf
import logging
import polars as pl


def __normalize(df: pl.DataFrame) -> pl.DataFrame:
    """
    Fix the dataframe returned by yf
    - convert index to column
    - convert date column to datetime
    """
    columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df.with_columns(columns)
    df = df.with_columns(
        df.select(pl.col("Date").str.to_datetime("%Y-%m-%d").alias("Date"))
    )
    return df


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

    logging.info("Fetching yfinance data for %s", ticker, start_date_str, end_date_str)

    data = yf.download(ticker, start=start_date_str, end=end_date_str)
    data = data.reset_index().rename(columns={"index": "Date"})
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")  # TODO

    df = __normalize(pl.from_pandas(data))
    return df
