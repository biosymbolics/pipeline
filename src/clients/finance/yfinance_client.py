"""
yfinance client
"""
from datetime import date
from typing import TypedDict
import yfinance as yf
import logging


StockPrice = TypedDict(
    "StockPrice", {"date": str, "close": float, "open": float, "volume": int}
)


def __normalize(data) -> list[StockPrice]:
    """
    Normalize data
    - turn date index into column of type str
    """
    try:
        data = data.reset_index().rename(columns={"index": "Date"})
        data["date"] = data["Date"].dt.strftime("%Y-%m-%d")
        data["close"] = data["Close"]
        data["open"] = data["Open"]
        data["volume"] = data["Volume"]
        df = data[["date", "close", "open", "volume"]]
        return df.to_dict("records")
    except Exception as e:
        logging.warning("Error normalizing data: %s (data: %s)", e, data)
        raise e


def fetch_yfinance_data(
    ticker: str, start_date: date, end_date: date
) -> list[StockPrice]:
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

    if len(data) > 0:
        return __normalize(data)

    return []
